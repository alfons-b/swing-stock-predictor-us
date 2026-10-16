"""Incremental ingestion (§10, §52).

DB latest date (per ticker) → expected market date → unduh HANYA yang hilang → validasi → UPSERT.
Tidak mengasumsikan run sebelumnya berhasil: titik awal selalu dihitung dari isi database, sehingga
run yang terlewat/gagal otomatis di-catch-up pada run berikutnya.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import get
from app.data.providers.chain import ProviderChain
from app.database.repository import Repository
from app.market.calendar import TradingCalendar
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class IngestStats:
    expected_date: str = ""
    index_rows: int = 0
    stocks_requested: int = 0
    stocks_up_to_date: int = 0
    stocks_processed: int = 0
    stocks_failed: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_rejected: int = 0
    refetched: list = field(default_factory=list)
    failed: dict = field(default_factory=dict)
    no_data: list = field(default_factory=list)
    provider_stats: dict = field(default_factory=dict)
    actions_inserted: int = 0
    backfill_rows: int = 0
    backfill_index_rows: int = 0

    def as_dict(self):
        d = self.__dict__.copy()
        d["failed"] = dict(list(self.failed.items())[:50])
        return d


def validate_new_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Validasi batch baru sebelum masuk DB: duplikat, missing, OHLC tidak valid, volume negatif."""
    n0 = len(df)
    d = df.drop_duplicates(["ticker", "date"], keep="last").dropna(subset=["open", "high", "low", "close", "volume"])
    bad = (d[["open", "high", "low", "close"]] <= 0).any(axis=1) | (d["high"] < d["low"]) | (d["volume"] < 0)
    d = d[~bad].copy()
    d["high"] = d[["open", "high", "low", "close"]].max(axis=1)
    d["low"] = d[["open", "high", "low", "close"]].min(axis=1)
    return d, n0 - len(d)


def initial_start(cfg: dict, cal: TradingCalendar) -> pd.Timestamp:
    s = get(cfg, "data.start_date")
    if s:
        return pd.Timestamp(s)
    return pd.Timestamp(cal.now().date()) - pd.DateOffset(years=int(get(cfg, "data.initial_history_years", 10)))


def ingest_index(cfg, repo: Repository, chain: ProviderChain, cal: TradingCalendar, expected) -> int:
    sym = get(cfg, "data.index_id", "COMPOSITE")
    last = repo.max_index_date(sym)
    if last is not None and last >= expected:
        return 0
    start = last + pd.Timedelta(days=1) if last is not None else initial_start(cfg, cal)
    df, src = chain.fetch_index(start, expected)
    df = df[(df["date"] >= start) & (df["date"] <= expected)].dropna(subset=["close"])
    df = df[df["close"] > 0]
    ins, _ = repo.upsert_index(df, sym, src)
    log.info("S&P 500: %d baris baru (%s) dari %s", ins, f"{start.date()}..{expected.date()}", src)
    return ins


def backfill_index(cfg, repo: Repository, chain: ProviderChain, cal: TradingCalendar) -> int:
    """Bila histori diperpanjang (initial_history_years/start_date diperbesar): isi S&P 500 di depan data yang ada."""
    sym = get(cfg, "data.index_id", "COMPOSITE")
    first_db = repo.min_index_date(sym)
    want = initial_start(cfg, cal)
    if first_db is None or first_db <= want + pd.Timedelta(days=7):
        return 0
    end = first_db - pd.Timedelta(days=1)
    df, src = chain.fetch_index(want, end)
    df = df[(df["date"] >= want) & (df["date"] <= end)].dropna(subset=["close"])
    ins, _ = repo.upsert_index(df[df["close"] > 0], sym, src)
    log.info("Backfill S&P 500: %d baris (%s..%s) dari %s", ins, want.date(), end.date(), src, extra={"persist": True})
    return ins


def backfill_prices(cfg, repo: Repository, chain: ProviderChain, cal: TradingCalendar) -> int:
    """Isi histori harga sebelum baris pertama tiap emiten, sampai initial_start. Idempoten (UPSERT).

    Dipanggil hanya saat histori S&P 500 baru diperpanjang, supaya emiten yang memang baru IPO
    tidak dicoba ulang setiap run.
    """
    want = initial_start(cfg, cal)
    latest = repo.latest_price_dates()
    latest = latest[(latest["n_rows"] > 0) & (latest["first_date"] > want + pd.Timedelta(days=10))]
    if latest.empty:
        return 0
    first = dict(zip(latest["ticker"], latest["first_date"]))
    log.info("Backfill histori %d emiten (%s s/d sebelum data pertama masing-masing)", len(first), want.date())
    res = chain.fetch_prices({t: want for t in first}, max(first.values()) - pd.Timedelta(days=1))
    ids = repo.stock_ids()
    max_jump = float(get(cfg, "data.split_refetch_jump", get(cfg, "quality.max_abs_daily_return", 0.35)))
    db_first_close = dict(repo.db.query(
        "SELECT s.ticker, p.close FROM price_history p JOIN stocks s ON s.id = p.stock_id "
        "JOIN (SELECT stock_id, MIN(date) AS d FROM price_history GROUP BY stock_id) m ON m.stock_id = p.stock_id AND m.d = p.date"))
    total = 0
    for t, f in res.frames.items():
        try:
            f, _ = validate_new_rows(f[(f["date"] >= want) & (f["date"] < first[t])])
            if f.empty:
                continue
            nxt = db_first_close.get(t)
            last_close = f.sort_values("date")["close"].iloc[-1]
            if nxt and res.adjusted.get(t) and abs(float(nxt) / last_close - 1) > max_jump:
                # skala harga berubah (split di antara dua unduhan) → muat ulang seluruh histori ticker ini
                full = chain.fetch_prices({t: want}, cal.expected_latest_market_date())
                if t not in full.frames:
                    continue
                f, _ = validate_new_rows(full.frames[t])
                repo.delete_prices(ids[t])
                log.warning("%s: skala harga backfill berbeda %.0f%% → histori dimuat ulang", t, (float(nxt) / last_close - 1) * 100)
            ins, _ = repo.upsert_prices(f.assign(stock_id=ids[t], is_adjusted=bool(res.adjusted.get(t)), source=res.sources[t]))
            total += ins
        except Exception as e:
            log.warning("Backfill %s gagal: %s", t, e)
    log.info("Backfill harga: +%d baris", total, extra={"persist": True})
    return total


def ingest_prices(cfg, repo: Repository, chain: ProviderChain, cal: TradingCalendar, expected,
                  tickers: list[str] | None = None) -> IngestStats:
    st = IngestStats(expected_date=str(pd.Timestamp(expected).date()))
    latest = repo.latest_price_dates().set_index("ticker")
    allst = repo.stocks()
    # Emiten aktif + emiten NONAKTIF yang belum punya histori (backfill s/d tanggal delisting).
    # Tanpa ini, saham yang sudah delisting tidak pernah masuk → survivorship bias di backtest.
    no_hist = allst["ticker"].map(lambda t: (latest.at[t, "n_rows"] if t in latest.index else 0) == 0)
    stocks = allst[allst["is_active"] | (no_hist & allst["delisting_date"].notna())]
    if tickers:
        stocks = stocks[stocks["ticker"].isin(tickers)]
    end_for = {r.ticker: min(pd.Timestamp(expected), pd.Timestamp(r.delisting_date)) if pd.notna(r.delisting_date) else pd.Timestamp(expected)
               for r in stocks.itertuples()}
    overlap = int(get(cfg, "data.overlap_days", 0))
    first = initial_start(cfg, cal)
    requests, last_close = {}, {}
    for t in stocks["ticker"]:
        last = latest.at[t, "last_date"] if t in latest.index else pd.NaT
        if pd.notna(last) and pd.Timestamp(last) >= end_for[t]:
            st.stocks_up_to_date += 1
            continue
        start = (pd.Timestamp(last) + pd.Timedelta(days=1) - pd.Timedelta(days=overlap)) if pd.notna(last) else first
        requests[t] = start
    st.stocks_requested = len(requests)
    if not requests:
        log.info("Semua %d emiten aktif sudah up-to-date (%s)", st.stocks_up_to_date, st.expected_date)
        return st
    # harga penutupan terakhir di DB → deteksi split yang belum tercatat
    if latest["n_rows"].fillna(0).sum() > 0:
        lc = repo.db.query_df("SELECT s.ticker, p.close FROM price_history p JOIN stocks s ON s.id = p.stock_id "
                              "JOIN (SELECT stock_id, MAX(date) AS d FROM price_history GROUP BY stock_id) m "
                              "ON m.stock_id = p.stock_id AND m.d = p.date")
        last_close = dict(zip(lc["ticker"], pd.to_numeric(lc["close"])))
    log.info("Mengunduh data hilang untuk %d emiten (s/d %s)", len(requests), st.expected_date)
    res = chain.fetch_prices(requests, expected)
    st.provider_stats = res.provider_stats
    ids = repo.stock_ids()
    max_jump = float(get(cfg, "data.split_refetch_jump", get(cfg, "quality.max_abs_daily_return", 0.35)))
    for t, f in res.frames.items():
        try:
            f = f[(f["date"] >= requests[t]) & (f["date"] <= end_for[t])]
            f, rejected = validate_new_rows(f)
            st.rows_rejected += rejected
            if f.empty:
                continue
            prev = last_close.get(t)
            first_close = f.sort_values("date")["close"].iloc[0]
            if prev and res.adjusted.get(t) and get(cfg, "data.refetch_on_split", True) and \
                    abs(first_close / prev - 1) > max_jump:
                # provider split-adjusted + lonjakan tak wajar → skala histori berubah: unduh ulang penuh
                full = chain.fetch_prices({t: first}, expected)
                if t in full.frames:
                    f, rej = validate_new_rows(full.frames[t])
                    st.rows_rejected += rej
                    repo.delete_prices(ids[t])
                    st.refetched.append(t)
                    log.warning("%s: lonjakan %.0f%% vs close terakhir → histori diunduh ulang (kemungkinan split)",
                                t, (first_close / prev - 1) * 100)
            rows = f.assign(stock_id=ids[t], is_adjusted=bool(res.adjusted.get(t)), source=res.sources[t])
            ins, upd = repo.upsert_prices(rows)
            st.rows_inserted += ins
            st.rows_updated += upd
            st.stocks_processed += 1
        except Exception as e:  # satu ticker gagal tidak membatalkan pipeline
            res.failed[t] = f"upsert: {type(e).__name__}: {e}"[:300]
    no_data = {t: m for t, m in res.failed.items() if t not in res.frames}
    quiet = sorted(t for t, m in no_data.items() if "tidak ada data baru" in m)
    if quiet:
        log.warning("%d emiten tanpa data dari provider (umumnya suspensi panjang / papan pemantauan khusus, atau belum "
                    "ada transaksi baru): %s%s", len(quiet), ", ".join(quiet[:40]), " ..." if len(quiet) > 40 else "")
    st.no_data = quiet
    # "tidak ada data baru" untuk ticker yang mungkin suspensi bukan kegagalan teknis
    st.failed = {t: m for t, m in no_data.items() if "tidak ada data baru" not in m}
    st.stocks_failed = len(st.failed)
    for prov in chain.providers:
        s = res.provider_stats.get(prov.name)
        if s:
            repo.source_status(prov.name, prov.type, prov.priority, True, ok=s["errors"] == 0 or s["ok"] > 0,
                               error=None if s["errors"] == 0 else f"{s['errors']} batch error", rows=s["rows"])
    return st


FULL_ACTIONS_SCAN = "corporate_actions_full_scan"


def ingest_actions(cfg, repo: Repository, chain: ProviderChain, tickers: list[str], start=None, end=None) -> int:
    """Corporate action per emiten. Provider umumnya butuh 1 request per emiten → diunduh paralel
    (market_data.action_workers); penulisan ke database tetap di thread utama."""
    if not tickers:
        return 0
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            return t, chain.fetch_actions(t, start, end)
        except Exception:
            return t, None

    workers = max(1, int(get(cfg, "market_data.action_workers", 4)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, tickers))
    ids = repo.stock_ids()
    frames = [a.assign(stock_id=ids[t], source="provider") for t, a in results if a is not None and not a.empty and t in ids]
    if not frames:
        return 0
    ins, _ = repo.upsert_actions(pd.concat(frames, ignore_index=True))
    return ins


def suspected_action_tickers(repo: Repository, since) -> list[str]:
    """Ticker dengan lonjakan harga harian > 25% sejak `since` → cek corporate action."""
    df = repo.load_prices(start=pd.Timestamp(since) - pd.Timedelta(days=10))
    if df.empty:
        return []
    r = df.groupby("ticker")["close"].pct_change().abs()
    return sorted(df.loc[(r > 0.25) & (df["date"] >= pd.Timestamp(since)), "ticker"].unique().tolist())


def run_ingestion(cfg, repo: Repository, chain: ProviderChain, cal: TradingCalendar,
                  full_actions: bool = False) -> IngestStats:
    expected = cal.expected_latest_market_date()
    st_index = ingest_index(cfg, repo, chain, cal, expected)
    back_index = backfill_index(cfg, repo, chain, cal)
    cal.learn_from_trading_days(repo.index_dates(get(cfg, "data.index_id", "COMPOSITE")))
    expected = cal.expected_latest_market_date()  # setelah belajar hari libur dari S&P 500
    before = repo.max_price_date()
    st = ingest_prices(cfg, repo, chain, cal, expected)
    st.index_rows, st.backfill_index_rows = st_index, back_index
    if back_index:  # histori diperpanjang → isi juga harga lama tiap emiten
        st.backfill_rows = backfill_prices(cfg, repo, chain, cal)
    active = repo.stocks(active_only=True)["ticker"].tolist()
    last_scan = repo.source_last_success(FULL_ACTIONS_SCAN)
    if last_scan is not None and last_scan.tzinfo is None:
        last_scan = last_scan.tz_localize("UTC")
    fresh_scan = last_scan is not None and \
        pd.Timestamp.now(tz="UTC") - last_scan < pd.Timedelta(days=int(get(cfg, "data.full_actions_scan_days", 7)))
    if (full_actions or before is None) and not fresh_scan:
        targets = active
        log.info("Cek corporate action seluruh %d emiten (paralel, %s worker)", len(active),
                 get(cfg, "market_data.action_workers", 4))
    else:
        targets = suspected_action_tickers(repo, before) if before is not None else []
    st.actions_inserted = ingest_actions(cfg, repo, chain, targets)
    if targets is active:
        repo.source_status(FULL_ACTIONS_SCAN, "job", 0, True, ok=True, rows=st.actions_inserted)
    return st
