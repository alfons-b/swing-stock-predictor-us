"""FlowProvider + ingest incremental ke `foreign_flow_history` (nama tabel dipertahankan dari versi BEI).

Pasar AS tidak mempublikasikan "foreign flow" (beli/jual investor asing) seperti BEI. Padanan harian GRATIS yang
tersedia untuk SELURUH saham AS adalah data **FINRA Reg SHO Daily Short Sale Volume**:
  https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt  (pipe-delimited, sejak 2018-08-01,
  dipublikasikan ≤ 18:00 ET pada hari yang sama). Kolom: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market.

Pemetaan ke skema kanonik (flow_normalizer.CANON) — dicatat di kolom `units`/`source` agar tidak disalahartikan:
  foreign_sell_shares = ShortVolume                (volume jual-short yang dilaporkan ke FINRA)
  foreign_buy_shares  = TotalVolume − ShortVolume  (volume non-short)
  total_volume_shares = TotalVolume                (volume off-exchange yang dilaporkan ke FINRA, BUKAN volume konsolidasi)
→ net = TotalVolume − 2·ShortVolume. Rasio short volume (SVR) struktural ±45–55% karena aktivitas market maker,
  sehingga fitur memakai SIMPANGAN terhadap baseline 120 hari bursa milik saham itu sendiri (flow_features,
  `foreign_flow.baseline_days`). Status NET_BUYING = tekanan short di bawah kebiasaannya; NET_SELLING = di atasnya.

Urutan sumber (config/research.yaml → foreign_flow.providers):
1. FINRAShortVolumeProvider — unduhan otomatis file harian FINRA (default, gratis).
2. CSVForeignFlowProvider   — CSV kanonik (ekspor vendor lain).
3. HTTPForeignFlowProvider  — API JSON pihak ketiga berlisensi (URL + API key dari GitHub Secrets).

Tanpa sumber AVAILABLE → status FOREIGN_FLOW_UNAVAILABLE (bukan nol).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.config import get, resolve_path
from app.database.repository import Repository, now_utc
from app.data.tickers import normalize_ticker
from app.flows.flow_normalizer import CANON, normalize
from app.utils.logging_utils import get_logger

log = get_logger(__name__)
_DATE_IN_NAME = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")


class ForeignFlowProvider:
    name = "base"

    def __init__(self, entry: dict, cfg: dict):
        self.entry, self.cfg = entry, cfg
        self.name = entry.get("name", self.name)
        self.priority = int(entry.get("priority", 50))

    def availability(self) -> tuple[str, str]:
        return "AVAILABLE", ""

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Data kanonik (lihat flow_normalizer.CANON) untuk start ≤ date ≤ end."""
        raise NotImplementedError


def file_date(path: Path) -> pd.Timestamp | None:
    m = _DATE_IN_NAME.search(path.stem)
    if not m:
        return None
    try:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    except ValueError:
        return None


class CSVForeignFlowProvider(ForeignFlowProvider):
    name = "csv_foreign_flow"

    def __init__(self, entry, cfg):
        super().__init__(entry, cfg)
        self.dir = resolve_path(cfg, str(entry.get("dir", "data/raw/foreign_flow_csv")))

    def _files(self):
        return sorted(self.dir.glob("*.csv")) if self.dir.exists() else []

    def availability(self):
        return ("AVAILABLE", f"{len(self._files())} file") if self._files() else \
            ("NOT_CONFIGURED", f"tidak ada CSV di {self.dir}")

    def fetch(self, start, end):
        frames = [normalize(pd.read_csv(p), self.name) for p in self._files()]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANON)
        return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


class HTTPForeignFlowProvider(ForeignFlowProvider):
    """API JSON berlisensi: GET {url}?date=YYYY-MM-DD, header Authorization: Bearer <api_key>.

    Respons: list record dengan kolom yang dikenali flow_normalizer (ticker, foreign_buy_shares, ...).
    Sesuaikan `_get` dengan kontrak vendor Anda. Kredensial hanya dari environment/GitHub Secrets.
    """
    name = "http_foreign_flow"

    def availability(self):
        if not self.entry.get("url") or not self.entry.get("api_key"):
            return "NOT_CONFIGURED", "FOREIGN_FLOW_API_URL / FOREIGN_FLOW_API_KEY belum diisi (GitHub Secrets)"
        return "AVAILABLE", "API pihak ketiga"

    def _get(self, day: pd.Timestamp) -> list[dict]:
        import requests
        r = requests.get(self.entry["url"], params={"date": day.strftime("%Y-%m-%d")},
                         headers={"Authorization": f"Bearer {self.entry['api_key']}"}, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("data", data) if isinstance(data, dict) else data

    def fetch(self, start, end):
        frames = []
        for day in pd.bdate_range(start, end):
            recs = self._get(day)
            if recs:
                df = pd.DataFrame(recs)
                df["date"] = df.get("date", day)
                frames.append(normalize(df, self.name))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANON)


class FINRAShortVolumeProvider(ForeignFlowProvider):
    """File harian FINRA Consolidated NMS (CNMS) short sale volume. Satu file = seluruh saham AS untuk 1 hari bursa.

    Hanya hari bursa (kalender NYSE) yang diunduh; hari tanpa file (libur / belum terbit) dilewati tanpa error.
    Unduhan pertama dibatasi `backfill_trading_days` agar setup tidak mengunduh bertahun-tahun file yang tidak muat
    di database paket gratis."""
    name = "finra_short_volume"
    URL = "https://cdn.finra.org/equity/regsho/daily/{prefix}shvol{ymd}.txt"
    FIRST_DATE = pd.Timestamp("2018-08-01")

    def __init__(self, entry, cfg):
        super().__init__(entry, cfg)
        self.prefix = str(entry.get("file_prefix", "CNMS"))
        self.backfill = int(entry.get("backfill_trading_days", 300))
        self.timeout = int(entry.get("timeout_seconds", 30))

    def availability(self):
        return "AVAILABLE", "FINRA Reg SHO daily short sale volume (cdn.finra.org, gratis)"

    @staticmethod
    def parse(text: str, day: pd.Timestamp | None = None) -> pd.DataFrame:
        """Isi file FINRA → kolom mentah (ticker, date, foreign_buy_shares, foreign_sell_shares, total_volume_shares)."""
        from io import StringIO
        lines = [ln for ln in text.splitlines() if "|" in ln]
        if len(lines) <= 1:
            return pd.DataFrame(columns=["ticker", "date"])
        df = pd.read_csv(StringIO("\n".join(lines)), sep="|", dtype=str)
        df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
        need = {"date", "symbol", "shortvolume", "totalvolume"}
        if not need <= set(df.columns):
            raise ValueError(f"format file FINRA berubah: kolom {list(df.columns)}")
        df = df[df["date"].str.fullmatch(r"\d{8}", na=False)]
        short = pd.to_numeric(df["shortvolume"], errors="coerce")
        total = pd.to_numeric(df["totalvolume"], errors="coerce")
        out = pd.DataFrame({"ticker": df["symbol"], "date": pd.to_datetime(df["date"], format="%Y%m%d"),
                            "foreign_buy_shares": (total - short).clip(lower=0), "foreign_sell_shares": short,
                            "total_volume_shares": total, "market_segment": "TOTAL",
                            "units": "shares (FINRA short volume: sell=short, buy=total-short)"})
        if day is not None:
            out = out[out["date"] == pd.Timestamp(day)]
        return out

    # Header mirip browser: CDN FINRA menolak (HTTP 403) User-Agent skrip generik dari IP datacenter. Di produksi
    # 16 Okt 2026 seluruh unduhan ditolak lalu dianggap "file belum terbit" → 0 baris tanpa error (diperbaiki di sini).
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/130.0 Safari/537.36",
               "Accept": "text/plain,text/html;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}

    def _download(self, day: pd.Timestamp) -> str | None:
        """Isi file, atau None bila belum terbit (404). HTTP 403/5xx dicatat sebagai DITOLAK (bukan 'belum terbit')."""
        import requests
        url = self.URL.format(prefix=self.prefix, ymd=day.strftime("%Y%m%d"))
        last = None
        for i in range(3):
            try:
                r = requests.get(url, timeout=self.timeout, headers=self.HEADERS)
            except Exception as e:
                last = e
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 403:
                self._blocked += 1
                return None
            if r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}")
                continue
            r.raise_for_status()
            self._ok += 1
            return r.text
        if last is not None:
            raise last
        return None

    def fetch(self, start, end):
        from app.market.calendar import TradingCalendar
        cal = TradingCalendar(self.cfg)
        end = pd.Timestamp(end).normalize()
        days = [d for d in pd.bdate_range(max(pd.Timestamp(start), self.FIRST_DATE), end) if cal.is_trading_day(d)]
        days = days[-self.backfill:]
        frames, missing = [], 0
        self._ok, self._blocked = 0, 0
        for d in days:
            txt = self._download(d)
            if self._blocked >= 5 and self._ok == 0:
                raise RuntimeError(f"CDN FINRA menolak akses (HTTP 403) untuk {self._blocked} file berturut-turut dari "
                                   "server ini — sumber flow tidak tersedia pada run ini")
            if not txt:
                missing += 1
                continue
            raw = self.parse(txt, d)
            if raw.empty:
                continue
            tickers = []
            for t in raw["ticker"]:
                try:
                    tickers.append(normalize_ticker(t))
                except ValueError:
                    tickers.append(None)
            raw["ticker"] = tickers
            raw = raw.dropna(subset=["ticker"])
            norm = normalize(raw, self.name, lot_size=1)
            norm["units"] = "shares (FINRA short volume: sell=short, buy=total-short)"
            norm["source_timestamp"] = d.strftime("%Y-%m-%d")
            frames.append(norm)
        if missing:
            log.info("FINRA: %d hari bursa tanpa file (%d ditolak 403, sisanya belum terbit/libur)", missing, self._blocked)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANON)


CLASSES = {"finra": FINRAShortVolumeProvider, "csv": CSVForeignFlowProvider, "http": HTTPForeignFlowProvider}


def make_flow_providers(cfg: dict) -> list[ForeignFlowProvider]:
    out = []
    for e in get(cfg, "foreign_flow.providers", []) or []:
        cls = CLASSES.get(e.get("type"))
        if cls is None:
            continue
        p = cls(e, cfg)
        p.enabled = str(e.get("enabled", True)).lower() in ("true", "1", "yes")
        out.append(p)
    return sorted(out, key=lambda p: p.priority)


FLOW_COLS = [c for c in CANON if c != "ticker"] + ["stock_id", "retrieved_at"]


def store_flows(repo: Repository, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    ids = repo.stock_ids()
    d = df.copy()
    d["stock_id"] = d["ticker"].map(ids)
    unknown = d["stock_id"].isna().sum()
    if unknown:
        log.debug("foreign flow: %d baris untuk emiten di luar universe dilewati", unknown)
    d = d.dropna(subset=["stock_id"])
    d["stock_id"] = d["stock_id"].astype(int)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["retrieved_at"] = now_utc()
    repo.db.upsert("foreign_flow_history", d.reindex(columns=FLOW_COLS), keys=["stock_id", "date", "market_segment"],
                   count=False)
    return len(d)


def update_foreign_flow(cfg: dict, repo: Repository, today, lookback_days: int | None = None, full: bool = False) -> dict:
    """Incremental: ambil dari tanggal terakhir tersimpan − lookback (koreksi data) s.d. hari ini."""
    today = pd.Timestamp(today).normalize()
    lookback_days = int(lookback_days if lookback_days is not None else get(cfg, "foreign_flow.lookback_days", 5))
    stats = {"providers": {}, "rows": 0, "status": "FOREIGN_FLOW_UNAVAILABLE"}
    if not get(cfg, "foreign_flow.enabled", True):
        stats["status"] = "DISABLED"
        return stats
    last = repo.db.scalar("SELECT MAX(date) FROM foreign_flow_history")
    start = pd.Timestamp("2000-01-01") if (full or not last) else pd.Timestamp(last) - pd.Timedelta(days=lookback_days)
    for p in make_flow_providers(cfg):
        status, detail = p.availability() if getattr(p, "enabled", True) else ("DISABLED", "enabled: false")
        stats["providers"][p.name] = {"status": status, "detail": detail}
        repo.set_source_availability(p.name, "foreign_flow", status, detail, p.priority)
        if status != "AVAILABLE":
            continue
        try:
            df = p.fetch(start, today)
        except Exception as e:
            stats["providers"][p.name] = {"status": "ERROR", "detail": str(e)[:300]}
            repo.set_source_availability(p.name, "foreign_flow", "ERROR", str(e), p.priority)
            log.warning("foreign flow %s gagal: %s", p.name, e)
            continue
        n = store_flows(repo, df)
        stats["rows"] += n
        stats["providers"][p.name]["rows"] = n
    if repo.db.scalar("SELECT COUNT(*) FROM foreign_flow_history WHERE date >= ?",
                      ((today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),)):
        stats["status"] = "AVAILABLE"
    elif repo.db.scalar("SELECT COUNT(*) FROM foreign_flow_history"):
        stats["status"] = "STALE"
    log.info("Foreign flow: %s (%d baris baru/diperbarui)", stats["status"], stats["rows"], extra={"persist": True})
    return stats


def load_flows(repo: Repository, start=None, end=None, segment: str | None = None) -> pd.DataFrame:
    sql = ("SELECT s.ticker, f.date, f.market_segment, f.foreign_buy_shares, f.foreign_sell_shares, f.net_foreign_shares, "
           "f.foreign_buy_value, f.foreign_sell_value, f.net_foreign_value, f.value_type, f.total_volume_shares, "
           "f.quality_status, f.source FROM foreign_flow_history f JOIN stocks s ON s.id = f.stock_id WHERE 1=1")
    params = []
    if start is not None:
        sql += " AND f.date >= ?"
        params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if end is not None:
        sql += " AND f.date <= ?"
        params.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
    if segment:
        sql += " AND f.market_segment = ?"
        params.append(segment)
    df = repo.db.query_df(sql, params, parse_dates=["date"])
    for c in ("foreign_buy_shares", "foreign_sell_shares", "net_foreign_shares", "foreign_buy_value", "foreign_sell_value",
              "net_foreign_value", "total_volume_shares"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df
