"""Membaca file daftar saham (CSV/XLSX) — format konstituen indeks AS atau format internal.

Format yang diterima (dideteksi otomatis dari nama kolom, huruf besar/kecil diabaikan):

1. Konstituen indeks (Wikipedia / datasets S&P 500 / ekspor screener):
       Symbol | Security | GICS Sector | GICS Sub-Industry | ...
2. Format internal:
       ticker, name, sector, subsector, listing_date, delisting_date, board

Hanya kolom kode saham (Symbol/ticker) yang wajib. `BRK.B` dan `BRK-B` dinormalkan ke `BRK-B`.
Sektor kosong diisi dari `universe.sectors_file` (bila ada) atau dari Yahoo secara bertahap.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.data.tickers import normalize_ticker

ALIASES = {
    "ticker": ["ticker", "symbol", "ticker symbol", "kode", "kode saham", "code", "stock code"],
    "name": ["name", "security", "company", "company name", "nama perusahaan", "nama"],
    "listing_date": ["listing_date", "listing date", "ipo date", "tanggal pencatatan"],
    "listed_shares": ["listed_shares", "shares outstanding", "shares"],
    "board": ["board", "index", "exchange", "listing exchange"],
    "sector": ["sector", "gics sector", "sektor"],
    "subsector": ["subsector", "gics sub-industry", "gics sub industry", "sub-industry", "industry", "subsektor"],
    "delisting_date": ["delisting_date", "tanggal delisting", "delisting date"],
}

MONTHS_ID = {"januari": "01", "jan": "01", "februari": "02", "feb": "02", "maret": "03", "mar": "03", "april": "04",
             "apr": "04", "mei": "05", "may": "05", "juni": "06", "jun": "06", "juli": "07", "jul": "07",
             "agustus": "08", "agu": "08", "agt": "08", "aug": "08", "september": "09", "sep": "09", "sept": "09",
             "oktober": "10", "okt": "10", "oct": "10", "november": "11", "nov": "11", "nop": "11",
             "desember": "12", "des": "12", "dec": "12"}


class UniverseFileError(ValueError):
    pass


EXAMPLE_FILE = "config/universe.example.csv"
CANDIDATES = ["config/universe.csv", "config/universe.xlsx"]


def resolve_universe_file(cfg: dict) -> tuple[Path | None, bool]:
    """(path, is_example). Urutan: universe.file → config/universe.xlsx|csv → daftar CONTOH.

    Daftar contoh hanya dipakai bila tidak ada file milik pengguna — dan selalu ditandai (health: WARN).
    File milik pengguna TIDAK ikut dikirim dalam zip/update proyek, sehingga tidak pernah tertimpa.
    """
    from app.config import get, resolve_path
    configured = get(cfg, "universe.file")
    paths = ([configured] if configured else []) + [c for c in CANDIDATES if c != configured]
    example = resolve_path(cfg, EXAMPLE_FILE)
    for rel in paths:
        p = resolve_path(cfg, rel)
        if p.exists():
            return p, is_example_list(p, example)
    return (example, True) if example.exists() else (None, False)


def is_example_list(path: Path, example: Path) -> bool:
    """True bila isi file sama dengan daftar contoh (mis. contoh yang tersalin sebagai universe.csv)."""
    if not example.exists() or path.resolve() == example.resolve():
        return example.exists()
    try:
        return set(load_universe_file(path)["ticker"]) == set(load_universe_file(example)["ticker"])
    except Exception:
        return False


def _norm(col) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower().replace("_", " ")).replace("listing date", "listing date")


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    for target, names in ALIASES.items():
        for n in names:
            lookup[_norm(n)] = target
    mapping = {}
    for c in df.columns:
        t = lookup.get(_norm(c))
        if t and t not in mapping.values():
            mapping[c] = t
    return df.rename(columns=mapping)


def parse_date(v):
    """Tanggal dalam berbagai format (termasuk nama bulan Indonesia). Tidak dikenali → NaT, bukan tebakan."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "-", "nan", "NaT"):
        return pd.NaT
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "year"):
        return pd.Timestamp(v).normalize()
    s = str(v).strip()
    if re.fullmatch(r"\d{5}(\.0+)?", s):  # nomor seri tanggal Excel
        return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(float(s)))).normalize()
    m = re.fullmatch(r"(\d{1,2})[\s\-/.]+([A-Za-z]+)[\s\-/.,]+(\d{4})", s)
    if m and m.group(2).lower() in MONTHS_ID:
        return pd.Timestamp(f"{m.group(3)}-{MONTHS_ID[m.group(2).lower()]}-{int(m.group(1)):02d}")
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}.*", s):
        return pd.to_datetime(s[:10], errors="coerce")
    return pd.to_datetime(s, dayfirst=True, errors="coerce")  # 20/05/2003 → 20 Mei, bukan 5 Okt


def parse_shares(v):
    """'15.000.000.000' (titik ribuan) / '15,000,000,000' / 1.5e10 → float."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    digits = re.sub(r"[^\d]", "", str(v))
    return float(digits) if digits else float("nan")


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm"):
        raw = pd.read_excel(path, dtype=object, header=None)
    elif suf == ".xls":
        try:
            raw = pd.read_excel(path, dtype=object, header=None)
        except ImportError as e:
            raise UniverseFileError("File .xls lama butuh paket xlrd — simpan ulang sebagai .xlsx atau CSV") from e
    elif suf in (".csv", ".txt", ".tsv"):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        first = text.splitlines()[0] if text else ""
        sep = "\t" if "\t" in first else ";" if first.count(";") > first.count(",") else ","
        raw = pd.read_csv(path, dtype=str, sep=sep, header=None, encoding="utf-8-sig")
    else:
        raise UniverseFileError(f"Format file universe tidak dikenal: {path.name} (pakai .xlsx atau .csv)")
    # baris judul bisa berada di bawah judul laporan → cari baris yang memuat kolom kode saham
    header_row = None
    for i in range(min(20, len(raw))):
        cells = {_norm(x) for x in raw.iloc[i].tolist() if pd.notna(x)}
        if cells & {_norm(n) for n in ALIASES["ticker"]}:
            header_row = i
            break
    if header_row is None:
        raise UniverseFileError(f"{path.name}: kolom kode saham ('Kode' / 'ticker') tidak ditemukan")
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    return df.dropna(how="all")


def load_universe_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = _rename(read_table(path))
    if "ticker" not in df:
        raise UniverseFileError(f"{path.name}: kolom kode saham tidak ditemukan")
    out = pd.DataFrame()
    bad = []
    tickers = []
    for t in df["ticker"]:
        try:
            tickers.append(normalize_ticker(t))
        except (ValueError, TypeError):
            tickers.append(None)
            if pd.notna(t) and str(t).strip():
                bad.append(str(t))
    out["ticker"] = tickers
    for col in ("name", "board", "sector", "subsector"):
        out[col] = df[col].astype(object).where(df[col].notna(), None).map(
            lambda x: str(x).strip() if x is not None else None).values if col in df else None
    out["listing_date"] = df["listing_date"].map(parse_date).values if "listing_date" in df else pd.NaT
    out["delisting_date"] = df["delisting_date"].map(parse_date).values if "delisting_date" in df else pd.NaT
    out["listed_shares"] = df["listed_shares"].map(parse_shares).values if "listed_shares" in df else float("nan")
    out = out[out["ticker"].notna()].drop_duplicates("ticker", keep="last").reset_index(drop=True)
    if out.empty:
        raise UniverseFileError(f"{path.name}: tidak ada kode saham yang valid")
    out.attrs["invalid_rows"] = bad[:20]
    return out


def apply_sectors(u: pd.DataFrame, sectors_path: Path | None) -> pd.DataFrame:
    """Gabungkan file sektor terpisah (ticker/Kode, sector/Sektor, subsector/Subsektor) bila tersedia."""
    if sectors_path is None or not Path(sectors_path).exists():
        return u
    s = _rename(read_table(Path(sectors_path)))
    if "ticker" not in s or "sector" not in s:
        raise UniverseFileError(f"{Path(sectors_path).name}: butuh kolom kode saham dan sektor")
    s["ticker"] = s["ticker"].map(lambda t: normalize_ticker(t) if pd.notna(t) else None)
    s = s.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").set_index("ticker")
    u = u.copy()
    for col in ("sector", "subsector"):
        if col in s:
            mapped = u["ticker"].map(s[col])
            u[col] = mapped.where(mapped.notna(), u.get(col))
    return u
