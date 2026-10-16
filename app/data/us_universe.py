"""Universe saham AS dari konstituen indeks (gratis, otomatis, membawa sektor GICS).

Sumber, berurutan:
1. Wikipedia — tabel konstituen "List of S&P 500 / 400 / 600 companies" dan "Nasdaq-100". Kolom yang dipakai:
   Symbol/Ticker, Security/Company, GICS Sector, GICS Sub-Industry. Diperbarui komunitas dalam hitungan jam/hari
   setelah perubahan indeks diumumkan S&P.
2. Dataset GitHub `datasets/s-and-p-500-companies` (CSV, diperbarui otomatis) — HANYA S&P 500.
3. `universe.file` (config/universe.csv di repo ini berisi snapshot S&P 500).

Pengaman deaktivasi: hasil hanya dianggap LENGKAP bila semua indeks di `universe.indexes` berhasil diambil dari
sumber utama. Hasil cadangan/parsial diberi `attrs["complete"] = False` dan UniverseManager TIDAK menonaktifkan
saham yang hilang darinya (mencegah ±400 saham mid-cap nonaktif hanya karena Wikipedia sedang gagal).

Catatan survivorship: universe berbasis indeks berarti saham yang KELUAR dari indeks (mis. turun ke S&P 600)
ditandai nonaktif mulai hari itu — historinya tetap ada di database, tidak dihapus.
"""
from __future__ import annotations

from io import StringIO

import pandas as pd

from app.config import get
from app.data.schema import conform_universe
from app.data.tickers import normalize_ticker
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

WIKI = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500", 400),
    "sp400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400", 300),
    "sp600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600", 450),
    "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Nasdaq-100", 90),
}
GITHUB_SP500 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
def user_agent() -> str:
    """Kebijakan User-Agent Wikimedia mewajibkan identitas + kontak; UA generik dari IP datacenter ditolak (HTTP 403).
    Di produksi 16 Okt 2026 S&P 400 gagal diambil sehingga universe hanya 503 saham. Kontak diambil dari
    SEC_USER_AGENT (yang memang berisi nama + email pemilik)."""
    import os
    contact = (os.environ.get("SEC_USER_AGENT") or "").strip()
    email = next((w for w in contact.split() if "@" in w), "")
    return f"swing-stock-predictor-us/1.1 (https://github.com/; {email or 'no-contact-configured'}) python-requests"


def _http_get(url: str, timeout: int = 30) -> str:
    """GET dengan cadangan: halaman wiki biasa → REST API HTML Wikipedia (endpoint berbeda, aturan cache berbeda)."""
    import requests
    headers = {"User-Agent": user_agent(), "Accept": "text/html,application/xhtml+xml"}
    urls = [url]
    if "wikipedia.org/wiki/" in url:
        title = url.split("/wiki/", 1)[1]
        urls.append(f"https://en.wikipedia.org/api/rest_v1/page/html/{title}")
    last = None
    for u in urls:
        try:
            r = requests.get(u, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
    raise last


def _pick(cols, *names):
    low = {str(c).strip().lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def constituents_from_table(df: pd.DataFrame, board: str) -> pd.DataFrame:
    """Tabel konstituen (Wikipedia/CSV) → kolom universe internal."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    sym = _pick(df.columns, "symbol", "ticker", "ticker symbol")
    if sym is None:
        raise ValueError("tabel tanpa kolom Symbol/Ticker")
    name = _pick(df.columns, "security", "company", "name")
    sector = _pick(df.columns, "gics sector", "sector", "gics sector[1]")
    sub = _pick(df.columns, "gics sub-industry", "gics sub industry", "sub-industry", "industry")
    rows = []
    def one(r, col):                                   # nama kolom ganda → ambil yang pertama
        if col is None:
            return None
        v = r[col]
        return v.iloc[0] if isinstance(v, pd.Series) else v

    for _, r in df.iterrows():
        try:
            t = normalize_ticker(one(r, sym))
        except (ValueError, TypeError):
            continue
        rows.append({"ticker": t, "name": one(r, name), "sector": one(r, sector), "subsector": one(r, sub),
                     "board": board})
    return pd.DataFrame(rows, columns=["ticker", "name", "sector", "subsector", "board"])


def parse_wikipedia(html: str, board: str, min_rows: int) -> pd.DataFrame:
    """Ambil tabel KONSTITUEN: kolom Symbol/Ticker tunggal + kolom sektor/nama, ≥ min_rows baris.

    Bug produksi 16 Okt 2026: halaman S&P 400 punya tabel riwayat perubahan indeks (Added/Removed, dua kolom
    "Ticker") yang lebih panjang dari tabel konstituen. Versi lama memilih tabel TERPANJANG → tabel riwayat →
    setiap baris gagal dibaca (dua kolom bernama sama) → 0 saham S&P 400 tanpa error, universe hanya 503.
    Sekarang: tabel riwayat ditolak, tabel dengan sektor GICS diutamakan, dan hasil akhir < min_rows = error."""
    tables = pd.read_html(StringIO(html))
    best, best_key = None, None
    for t in tables:
        multi = isinstance(t.columns, pd.MultiIndex)
        top = [str(c[0]).strip().lower() for c in t.columns] if multi else []
        cols = [str(c[-1] if isinstance(c, tuple) else c).strip().lower() for c in t.columns]
        n_sym = sum(c in ("symbol", "ticker", "ticker symbol") for c in cols)
        if n_sym != 1 or {"added", "removed"} & set(top):
            continue                                   # bukan tabel konstituen (mis. riwayat perubahan indeks)
        has_sector = any("sector" in c for c in cols)
        has_name = bool({"security", "company", "name"} & set(cols))
        key = (has_sector, has_name, len(t))
        if best_key is None or key > best_key:
            best, best_key = t, key
    if best is None or len(best) < min_rows:
        raise ValueError(f"tabel konstituen {board} tidak ditemukan / terlalu pendek "
                         f"({0 if best is None else len(best)} < {min_rows} baris)")
    out = constituents_from_table(best, board)
    if len(out) < min_rows:
        raise ValueError(f"konstituen {board} hanya {len(out)} ticker valid dari {len(best)} baris (< {min_rows})")
    return out


class USIndexUniverse:
    type = "us_index"
    name = "us_index"

    def __init__(self, cfg: dict, http_get=_http_get):
        self.cfg = cfg
        self.http_get = http_get
        self.indexes = [str(i).lower() for i in (get(cfg, "universe.indexes", ["sp500", "sp400"]) or ["sp500"])]

    def get_universe(self) -> pd.DataFrame:
        frames, missing, used, errors = [], [], [], {}
        for ix in self.indexes:
            if ix not in WIKI:
                log.warning("Indeks universe tidak dikenal: %s (pilihan: %s)", ix, ", ".join(WIKI))
                continue
            url, board, min_rows = WIKI[ix]
            try:
                frames.append(parse_wikipedia(self.http_get(url), board, min_rows))
                used.append(f"wikipedia:{ix}")
            except Exception as e:
                log.warning("Konstituen %s dari Wikipedia gagal: %s", board, e)
                missing.append(ix)
                errors[ix] = f"{type(e).__name__}: {e}"[:300]
        if "sp500" in missing:
            try:
                frames.append(constituents_from_table(pd.read_csv(StringIO(self.http_get(GITHUB_SP500))), "S&P 500"))
                used.append("github:sp500")
                log.info("S&P 500 diambil dari dataset GitHub (cadangan)")
            except Exception as e:
                log.warning("Konstituen S&P 500 dari GitHub gagal: %s", e)
        if not frames:
            raise RuntimeError("Semua sumber konstituen indeks gagal")
        u = pd.concat(frames, ignore_index=True).drop_duplicates("ticker", keep="first")
        u = conform_universe(u)
        u.attrs["complete"] = not missing
        u.attrs["source_detail"] = ",".join(used)
        u.attrs["errors"] = errors
        u.attrs["counts"] = u["board"].value_counts().to_dict()
        if missing:
            log.warning("Universe PARSIAL (gagal: %s) — saham yang tidak ada di daftar ini TIDAK dinonaktifkan",
                        ", ".join(missing))
        return u
