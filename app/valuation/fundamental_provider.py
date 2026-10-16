"""FundamentalDataProvider (§B3) + penyimpanan point-in-time ke `financial_statements`.

Prioritas sumber (config/research.yaml → fundamentals.providers):
1. SECEdgarFundamentalProvider — laporan 10-K/10-Q resmi dari SEC EDGAR (gratis, point-in-time dari tanggal filed;
   app/valuation/sec_edgar.py). Butuh env SEC_USER_AGENT ("Nama email@domain.com").
2. CSVFundamentalProvider   — file kanonik (ekspor dari laporan resmi / provider berlisensi), dengan tanggal publikasi.
3. YahooFundamentalProvider — snapshot ±4 tahun + ±5 kuartal TERAKHIR. Tidak punya tanggal publikasi dan bisa berisi
   angka restated → `first_known_date` = tanggal sistem mengambilnya (aman untuk live, tidak untuk backtest lampau).

Tidak ada sumber yang mengisi angka kosong dengan nol. Data tak tersedia = None + status UNAVAILABLE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import get, resolve_path
from app.data.tickers import normalize_ticker, to_provider_symbol
from app.database.repository import Repository, now_utc
from app.utils.logging_utils import get_logger
from app.valuation.financial_statement_normalizer import CANONICAL, Statement, normalize_items, quality_checks

log = get_logger(__name__)


class FundamentalUnavailable(RuntimeError):
    pass


def quarter_label(period_end: pd.Timestamp, fy_end_month: int = 12) -> str:
    m = (pd.Timestamp(period_end).month - fy_end_month - 1) % 12 + 1   # bulan ke-n tahun fiskal
    return f"Q{int(np.ceil(m / 3))}"


class FundamentalDataProvider:
    name = "base"
    #: True bila sumber memberi tanggal publikasi asli & angka sebagaimana dipublikasikan (bukan restated)
    point_in_time = False

    def __init__(self, entry: dict, cfg: dict):
        self.entry, self.cfg = entry, cfg
        self.name = entry.get("name", self.name)
        self.priority = int(entry.get("priority", 50))

    def availability(self) -> tuple[str, str]:
        """(status, detail): AVAILABLE | UNAVAILABLE | NOT_CONFIGURED."""
        return "AVAILABLE", ""

    def get_statements(self, ticker: str) -> list[Statement]:
        raise NotImplementedError


class CSVFundamentalProvider(FundamentalDataProvider):
    """File kanonik: data/raw/fundamentals/*.csv|xlsx.

    Kolom wajib: ticker, period_end, period_type (FY/Q1..Q4), currency. Opsional: publication_date, source,
    dan field kanonik apa pun (revenue, net_income, total_equity, ...). Nilai dalam SATUAN PENUH mata uang laporan.
    """
    name = "csv_fundamentals"

    def __init__(self, entry, cfg):
        super().__init__(entry, cfg)
        self.dir = resolve_path(cfg, str(entry.get("dir", "data/raw/fundamentals")))
        self.point_in_time = bool(entry.get("point_in_time", True))
        self._df = None

    def _files(self):
        return sorted(list(self.dir.glob("*.csv")) + list(self.dir.glob("*.xlsx"))) if self.dir.exists() else []

    def availability(self):
        if not self._files():
            return "NOT_CONFIGURED", f"tidak ada file di {self.dir}"
        return "AVAILABLE", f"{len(self._files())} file"

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            frames = [pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p) for p in self._files()]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if len(df):
                df.columns = [str(c).strip().lower() for c in df.columns]
                missing = {"ticker", "period_end", "period_type", "currency"} - set(df.columns)
                if missing:
                    raise FundamentalUnavailable(f"file fundamental tanpa kolom wajib: {sorted(missing)}")
                df["ticker"] = df["ticker"].map(normalize_ticker)
            self._df = df
        return self._df

    def get_statements(self, ticker):
        df = self._load()
        if df.empty:
            return []
        rows = df[df["ticker"] == normalize_ticker(ticker)]
        out, now = [], now_utc()
        for r in rows.to_dict("records"):
            items = {k: (None if pd.isna(r.get(k)) else float(r[k])) for k in CANONICAL if k in r}
            items = {**{k: None for k in CANONICAL}, **items}
            if items.get("free_cash_flow") is None and items.get("operating_cash_flow") is not None and items.get("capex") is not None:
                items["free_cash_flow"] = items["operating_cash_flow"] - abs(items["capex"])
            pub = pd.Timestamp(r["publication_date"]) if pd.notna(r.get("publication_date")) else None
            out.append(Statement(ticker=normalize_ticker(ticker), period_end=pd.Timestamp(r["period_end"]),
                                 period_type=str(r["period_type"]).upper(), currency=str(r["currency"]).upper(),
                                 items=items, source=str(r.get("source") or self.name), retrieved_at=now, publication_date=pub))
        return out


class YahooFundamentalProvider(FundamentalDataProvider):
    """yfinance: income_stmt / balance_sheet / cashflow (tahunan & kuartalan) + info (mata uang, jumlah saham)."""
    name = "yahoo_fundamentals"

    def __init__(self, entry, cfg):
        super().__init__(entry, cfg)
        self.suffix = entry.get("suffix", "")

    def availability(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return "UNAVAILABLE", "paket yfinance belum terpasang"
        return "AVAILABLE", "snapshot (bukan point-in-time historis; angka bisa restated)"

    @staticmethod
    def _frame_to_periods(df) -> dict[pd.Timestamp, dict]:
        if df is None or getattr(df, "empty", True):
            return {}
        out = {}
        for col in df.columns:
            out[pd.Timestamp(col).normalize()] = {str(idx): df.at[idx, col] for idx in df.index}
        return out

    def get_statements(self, ticker):
        import yfinance as yf
        t = yf.Ticker(to_provider_symbol(ticker, self.suffix))
        try:
            info = t.info or {}
        except Exception:
            info = {}
        currency = info.get("financialCurrency") or info.get("currency")
        now = now_utc()
        out = []
        for ptype, parts in (("FY", (t.income_stmt, t.balance_sheet, t.cashflow)),
                             ("Q", (t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow))):
            merged: dict[pd.Timestamp, dict] = {}
            for part in parts:
                for pe, raw in self._frame_to_periods(part).items():
                    merged.setdefault(pe, {}).update(raw)
            for pe, raw in merged.items():
                items = normalize_items(raw)
                if all(items[k] is None for k in ("revenue", "net_income", "total_equity", "operating_cash_flow")):
                    continue
                out.append(Statement(ticker=normalize_ticker(ticker), period_end=pe,
                                     period_type="FY" if ptype == "FY" else quarter_label(pe), currency=currency,
                                     items=items, source=self.name, retrieved_at=now))
        return out


def make_fundamental_providers(cfg: dict) -> list[FundamentalDataProvider]:
    from app.valuation.sec_edgar import SECEdgarFundamentalProvider
    classes = {"sec_edgar": SECEdgarFundamentalProvider, "csv": CSVFundamentalProvider, "yahoo": YahooFundamentalProvider}
    out = []
    for e in get(cfg, "fundamentals.providers", []) or []:
        if str(e.get("enabled", True)).lower() not in ("true", "1", "yes"):
            continue
        cls = classes.get(e.get("type"))
        if cls:
            out.append(cls(e, cfg))
    return sorted(out, key=lambda p: p.priority)


# ------------------------------------------------------------------------------------------ penyimpanan PIT
def estimated_available(period_end: pd.Timestamp, period_type: str, cfg: dict) -> pd.Timestamp:
    lag = get(cfg, "fundamentals.estimated_publication_lag_days", {}) or {}
    days = int(lag.get("FY", 150) if period_type == "FY" else lag.get("Q", 100))
    return pd.Timestamp(period_end) + pd.Timedelta(days=days)


def store_statements(repo: Repository, cfg: dict, statements: list[Statement], today: pd.Timestamp,
                     point_in_time_source: bool) -> int:
    """UPSERT idempoten. Angka berubah untuk periode yang sama (restatement) → versi baru, versi lama tetap ada."""
    if not statements:
        return 0
    ids = repo.stock_ids()
    hashes = repo.db.query_df("SELECT stock_id, period_end, period_type, source, version, items_hash FROM financial_statements")
    latest_hash = {}
    if len(hashes):
        hashes["period_end"] = pd.to_datetime(hashes["period_end"]).dt.strftime("%Y-%m-%d")
        hashes = hashes.sort_values("version")
        for r in hashes.itertuples():
            latest_hash[(int(r.stock_id), r.period_end, r.period_type, r.source)] = (int(r.version), r.items_hash)
    rows = []
    price_ccy = get(cfg, "fundamentals.price_currency", "USD")
    for st in statements:
        sid = ids.get(st.ticker)
        if sid is None:
            continue
        quality_checks(st, price_ccy)
        pe = pd.Timestamp(st.period_end).strftime("%Y-%m-%d")
        key = (sid, pe, st.period_type, st.source)
        h = st.items_hash()
        prev = latest_hash.get(key)
        if prev and prev[1] == h:
            continue                                         # tidak berubah → tidak ditulis ulang
        version = (prev[0] + 1) if prev else 1
        known = today
        if point_in_time_source and st.publication_date is not None and pd.Timestamp(st.publication_date) <= today:
            known = pd.Timestamp(st.publication_date)
        if prev:                                             # restatement baru diketahui hari ini
            known = today
        rows.append({"stock_id": sid, "period_end": pe, "period_type": st.period_type, "fiscal_year": st.fiscal_year,
                     "currency": st.currency, "items": json.dumps(st.items), "items_hash": h,
                     "publication_date": None if st.publication_date is None else pd.Timestamp(st.publication_date).strftime("%Y-%m-%d"),
                     "first_known_date": known.strftime("%Y-%m-%d"),
                     "estimated_available_date": estimated_available(st.period_end, st.period_type, cfg).strftime("%Y-%m-%d"),
                     "retrieved_at": st.retrieved_at, "source": st.source, "version": version,
                     "quality_status": st.quality_status, "quality_notes": "; ".join(st.quality_notes)[:2000]})
    if rows:
        repo.db.upsert("financial_statements", pd.DataFrame(rows),
                       keys=["stock_id", "period_end", "period_type", "source", "version"], count=False)
    return len(rows)


def store_earnings_events(repo: Repository, ev: pd.DataFrame) -> int:
    """UPSERT idempoten per (stock_id, accession)."""
    if ev is None or ev.empty:
        return 0
    ids = repo.stock_ids()
    d = ev.copy()
    d["stock_id"] = d["ticker"].map(ids)
    d = d.dropna(subset=["stock_id", "reaction_date"])
    if d.empty:
        return 0
    d["stock_id"] = d["stock_id"].astype(int)
    for c in ("filing_date", "reaction_date"):
        d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d")
    d["accepted_at"] = d["accepted_at"].astype(object).where(d["accepted_at"].notna(), None).map(
        lambda x: None if x is None else str(x))
    d["source"], d["retrieved_at"] = "sec_8k_2.02", now_utc()
    repo.db.upsert("earnings_events", d[["stock_id", "accession", "filing_date", "accepted_at", "reaction_date", "form",
                                         "source", "retrieved_at"]], keys=["stock_id", "accession"], count=False)
    return len(d)


def load_earnings_events(repo: Repository) -> pd.DataFrame:
    return repo.db.query_df("SELECT s.ticker, e.reaction_date FROM earnings_events e JOIN stocks s ON s.id = e.stock_id",
                            parse_dates=["reaction_date"])


def earnings_watchlist(events: pd.DataFrame, today, before_days: int = 3, after_days: int = 12) -> list[str]:
    """Ticker yang rilis labanya diperkirakan jatuh di [today − after_days, today + before_days] dan belum tercatat.

    Mengapa: laporan fundamental tiap saham hanya dicek ulang tiap `refresh_days` (7 hari), sehingga 8-K laba baru
    bisa terlambat terdeteksi hingga seminggu — sementara backtest mengasumsikan event diketahui H+1. Daftar ini
    dicek SETIAP hari (hanya 1 request SEC per ticker) agar fitur PEAD/earnings di scan live = di backtest."""
    if events is None or events.empty:
        return []
    today = pd.Timestamp(today).normalize()
    out = []
    for t, g in events.groupby("ticker"):
        d = pd.to_datetime(g["reaction_date"]).sort_values().drop_duplicates()
        last = d.iloc[-1]
        gaps = d.diff().dt.days.dropna()
        gaps = gaps[(gaps >= 45) & (gaps <= 140)].tail(4)
        if gaps.empty or (today - last).days < 30:          # baru rilis / tanpa pola kuartalan → tidak perlu dipantau
            continue
        est = last + pd.Timedelta(days=int(gaps.median()))
        if est - pd.Timedelta(days=before_days) <= today <= est + pd.Timedelta(days=after_days):
            out.append(t)
    return out


def load_statements(repo: Repository, as_of=None, use_estimated: bool = False) -> pd.DataFrame:
    """Laporan yang SUDAH diketahui pada `as_of` (versi terakhir per periode & sumber).

    use_estimated=True (riset backtest saja) memakai estimated_available_date untuk data yang first_known_date-nya
    setelah as_of — hasilnya harus diberi label ESTIMATED_PIT.
    """
    df = repo.db.query_df("SELECT f.*, s.ticker FROM financial_statements f JOIN stocks s ON s.id = f.stock_id",
                          parse_dates=["period_end", "first_known_date", "estimated_available_date", "publication_date"])
    if df.empty:
        return df
    df["usable_from"] = df["first_known_date"]
    if use_estimated:
        df["usable_from"] = df[["first_known_date", "estimated_available_date"]].min(axis=1)
    if as_of is not None:
        df = df[df["usable_from"] <= pd.Timestamp(as_of)]
    df = df.sort_values("version").groupby(["stock_id", "period_end", "period_type", "source"], as_index=False).last()
    df["items"] = df["items"].map(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
    return df


def update_fundamentals(cfg: dict, repo: Repository, today, tickers: list[str] | None = None,
                        earnings_backfill: bool = False) -> dict:
    """Incremental: hanya emiten yang belum pernah dicek / sudah > refresh_days, maksimal max_per_run per run.
    earnings_backfill=True: juga ambil riwayat 8-K laba untuk SEMUA saham aktif yang belum punya event (dipakai
    workflow Fundamental Refresh; wajib dijalankan sekali setelah upgrade ke fitur f2, sebelum Daily)."""
    today = pd.Timestamp(today).normalize()
    providers = make_fundamental_providers(cfg)
    stats = {"providers": {}, "checked": 0, "new_versions": 0, "failed": 0}
    usable = []
    for p in providers:
        status, detail = p.availability()
        stats["providers"][p.name] = {"status": status, "detail": detail}
        repo.set_source_availability(p.name, "fundamental", status, detail, p.priority)
        if status == "AVAILABLE":
            usable.append(p)
    if not usable:
        return stats
    st = repo.stocks(active_only=True)
    refresh = pd.Timedelta(days=int(get(cfg, "fundamentals.refresh_days", 7)))
    checked = pd.to_datetime(st["fundamentals_checked_at"], errors="coerce", utc=True) if "fundamentals_checked_at" in st \
        else pd.Series(pd.NaT, index=st.index)
    now = pd.Timestamp.now(tz="UTC")
    # assign SEBELUM filter: assign ke frame kosong mengadopsi index Series → semua saham dianggap jatuh tempo
    due = st.assign(_c=checked)[checked.isna() | (now - checked > refresh)].sort_values("_c", na_position="first")
    if tickers:
        due = st[st["ticker"].isin(tickers)]
    due = due.head(int(get(cfg, "fundamentals.max_per_run", 150)))
    for t in due["ticker"]:
        got = False
        for p in usable:
            try:
                sts = p.get_statements(t)
            except Exception as e:
                log.debug("fundamental %s via %s gagal: %s", t, p.name, e)
                continue
            if sts:
                stats["new_versions"] += store_statements(repo, cfg, sts, today, p.point_in_time)
                got = True
                break
        for p in usable:                                  # tanggal rilis laba (SEC 8-K 2.02) — fitur & filter earnings
            if hasattr(p, "get_earnings_events"):
                try:
                    stats["earnings_events"] = stats.get("earnings_events", 0) + store_earnings_events(
                        repo, p.get_earnings_events(t))
                except Exception as e:
                    log.debug("earnings %s via %s gagal: %s", t, p.name, e)
                break
        stats["checked"] += 1
        stats["failed"] += 0 if got else 1
        repo.db.execute("UPDATE stocks SET fundamentals_checked_at = ? WHERE ticker = ?", (now_utc(), t))
    ep = next((p for p in usable if hasattr(p, "get_earnings_events")), None)
    if ep is not None and not tickers:
        try:
            ev = load_earnings_events(repo)
        except Exception:                                  # tabel belum ada (skema lama)
            ev = pd.DataFrame()
        active = set(st["ticker"])
        done = set(due["ticker"])
        watch = [t for t in earnings_watchlist(ev, today) if t in active and t not in done]
        watch = watch[:int(get(cfg, "fundamentals.earnings_watch_max", 300) or 0)]
        found = 0
        for t in watch:
            try:
                found += store_earnings_events(repo, ep.get_earnings_events(t, years=1))
            except Exception as e:
                log.debug("earnings-watch %s gagal: %s", t, e)
        stats["earnings_watch"] = {"checked": len(watch), "rows": found}
        if earnings_backfill:
            have = set(ev["ticker"]) if len(ev) else set()
            todo = [t for t in st["ticker"] if t not in have and t not in done]
            found = 0
            for t in todo:
                try:
                    found += store_earnings_events(repo, ep.get_earnings_events(t))
                except Exception as e:
                    log.debug("earnings-backfill %s gagal: %s", t, e)
            stats["earnings_backfill"] = {"checked": len(todo), "rows": found}
            log.info("Earnings backfill: %d saham tanpa riwayat dicek, %d event tersimpan", len(todo), found,
                     extra={"persist": True})
    log.info("Fundamental: %d emiten dicek, %d versi laporan baru, %d tanpa data", stats["checked"], stats["new_versions"],
             stats["failed"], extra={"persist": True})
    return stats


def csv_template(path: Path) -> Path:
    """Template file fundamental kanonik untuk diisi dari laporan resmi."""
    cols = ["ticker", "period_end", "period_type", "currency", "publication_date", "source"] + CANONICAL
    pd.DataFrame(columns=cols).to_csv(path, index=False)
    return path
