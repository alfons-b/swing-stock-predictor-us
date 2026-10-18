"""Penyimpanan point-in-time + penjadwal kuota Alpha Vantage.

Satu run (`python main.py av-update --mode daily|backfill`) menghabiskan kuota 24 jam yang tersisa dengan urutan
nilai informasi tertinggi lebih dulu:

  daily    : kalender rilis (1) → yield Treasury yang basi (0–3) → berita (2) → surprise EPS (sisanya)
  backfill : surprise EPS saja (workflow akhir pekan)

Antrean surprise EPS (EARNINGS, 1 request = seluruh riwayat kuartalan satu saham):
  1. SEGAR  — saham yang baru rilis laba (8-K SEC atau kalender ≤ fresh_report_days hari) tetapi surprise kuartal
              itu belum tersimpan. Sinyal PEAD paling kuat di minggu-minggu awal setelah rilis.
  2. BARU   — saham yang belum pernah diambil (backfill; S&P 500 dulu, lalu S&P 400).
  3. BASI   — sisanya, yang paling lama tidak dicek (> earnings_refresh_days).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from app.config import get
from app.data.alpha_vantage import AlphaVantageClient, AlphaVantageError, PremiumOnly, QuotaExhausted
from app.utils.logging_utils import get_logger

log = get_logger(__name__)
SOURCE = "alpha_vantage"


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def _d(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


# ------------------------------------------------------------------------------------------------ simpan / muat
def store_surprises(repo, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    ids = repo.stock_ids()
    d = df.copy()
    d["stock_id"] = d["ticker"].map(ids)
    d = d.dropna(subset=["stock_id", "fiscal_date_ending", "reported_date"])
    if d.empty:
        return 0
    d["stock_id"] = d["stock_id"].astype(int)
    d["fiscal_date_ending"], d["reported_date"] = _d(d["fiscal_date_ending"]), _d(d["reported_date"])
    d["source"], d["retrieved_at"] = SOURCE, _now()
    cols = ["stock_id", "fiscal_date_ending", "reported_date", "report_time", "reported_eps", "estimated_eps",
            "surprise", "surprise_pct", "source", "retrieved_at"]
    d = d[cols].astype(object).where(d[cols].notna(), None)
    repo.db.upsert("earnings_surprises", d, keys=["stock_id", "fiscal_date_ending"], count=False)
    return len(d)


def _obj_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) and "ticker" in df:
        df["ticker"] = df["ticker"].astype(object)
    return df


def load_surprises(repo) -> pd.DataFrame:
    return _obj_ticker(repo.db.query_df("SELECT s.ticker, e.fiscal_date_ending, e.reported_date, e.report_time, e.reported_eps, "
                            "e.estimated_eps, e.surprise, e.surprise_pct FROM earnings_surprises e "
                            "JOIN stocks s ON s.id = e.stock_id", parse_dates=["fiscal_date_ending", "reported_date"]))


def store_calendar(repo, cal: pd.DataFrame, as_of) -> int:
    """Log perubahan: simpan baris hanya bila (tanggal rilis, estimasi, jam) berbeda dari versi terakhir."""
    if cal is None or cal.empty:
        return 0
    ids = repo.stock_ids()
    d = cal.copy()
    d["stock_id"] = d["ticker"].map(ids)
    d = d.dropna(subset=["stock_id", "report_date"])
    if d.empty:
        return 0
    d["stock_id"] = d["stock_id"].astype(int)
    d["fiscal_date_ending"] = d["fiscal_date_ending"].fillna(d["report_date"])
    d["fiscal_date_ending"], d["report_date"] = _d(d["fiscal_date_ending"]), _d(d["report_date"])
    d = d.drop_duplicates(["stock_id", "fiscal_date_ending"], keep="first")
    prev = repo.db.query_df(
        "SELECT c.stock_id, c.fiscal_date_ending, c.report_date, c.estimate, c.time_of_day FROM earnings_calendar c "
        "JOIN (SELECT stock_id, fiscal_date_ending, MAX(as_of) AS m FROM earnings_calendar GROUP BY stock_id, "
        "fiscal_date_ending) x ON x.stock_id = c.stock_id AND x.fiscal_date_ending = c.fiscal_date_ending AND x.m = c.as_of")
    if len(prev):
        prev["fiscal_date_ending"], prev["report_date"] = _d(prev["fiscal_date_ending"]), _d(prev["report_date"])
        m = d.merge(prev, on=["stock_id", "fiscal_date_ending"], how="left", suffixes=("", "_p"))
        est_same = (m["estimate"].round(4) == pd.to_numeric(m["estimate_p"], errors="coerce").round(4)) | \
                   (m["estimate"].isna() & m["estimate_p"].isna())
        same = (m["report_date"] == m["report_date_p"]) & est_same & \
               (m["time_of_day"].fillna("") == m["time_of_day_p"].fillna(""))
        d = m.loc[~same, d.columns]
    if d.empty:
        return 0
    d["as_of"], d["retrieved_at"] = pd.Timestamp(as_of).strftime("%Y-%m-%d"), _now()
    cols = ["stock_id", "fiscal_date_ending", "report_date", "estimate", "time_of_day", "as_of", "retrieved_at"]
    d = d[cols].astype(object).where(d[cols].notna(), None)
    repo.db.upsert("earnings_calendar", d, keys=["stock_id", "fiscal_date_ending", "as_of"], count=False)
    return len(d)


def load_calendar(repo) -> pd.DataFrame:
    return _obj_ticker(repo.db.query_df("SELECT s.ticker, c.fiscal_date_ending, c.report_date, c.estimate, c.time_of_day, c.as_of "
                            "FROM earnings_calendar c JOIN stocks s ON s.id = c.stock_id",
                            parse_dates=["fiscal_date_ending", "report_date", "as_of"]))


def store_series(repo, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    d = df.copy()
    d["date"] = _d(d["date"])
    d["source"], d["retrieved_at"] = SOURCE, _now()
    repo.db.upsert("macro_series", d[["series", "date", "value", "source", "retrieved_at"]], keys=["series", "date"],
                   count=False)
    return len(d)


def load_series(repo, series: str | None = None) -> pd.DataFrame:
    q, p = "SELECT series, date, value FROM macro_series", ()
    if series:
        q, p = q + " WHERE series = ?", (series,)
    return repo.db.query_df(q, p, parse_dates=["date"])


def latest_series(repo, series: str, upto=None):
    q, p = "SELECT date, value FROM macro_series WHERE series = ?", [series]
    if upto is not None:                                   # point-in-time: nilai yang sudah terbit pada `upto`
        q, p = q + " AND date <= ?", p + [pd.Timestamp(upto).strftime("%Y-%m-%d")]
    r = repo.db.query(q + " ORDER BY date DESC LIMIT 1", tuple(p))
    return (pd.Timestamp(r[0][0]), float(r[0][1])) if r else (None, None)


def store_news(repo, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    ids = repo.stock_ids()
    d = df.copy()
    d["stock_id"] = d["ticker"].map(ids)
    d = d.dropna(subset=["stock_id"])
    if d.empty:
        return 0
    d["stock_id"] = d["stock_id"].astype(int)
    d["url_hash"] = (d["url"].fillna("") + "|" + d["headline"].fillna("")).map(
        lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    d["published_at"] = pd.to_datetime(d["published_at"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    d["headline"] = d["headline"].str.slice(0, 160)
    d["retrieved_at"] = _now()
    d = d.drop_duplicates(["stock_id", "url_hash"])
    cols = ["stock_id", "published_at", "url_hash", "headline", "source", "sentiment", "relevance", "retrieved_at"]
    repo.db.upsert("news_items", d[cols], keys=["stock_id", "url_hash"], count=False)
    return len(d)


def load_news(repo, since=None) -> pd.DataFrame:
    q, p = ("SELECT s.ticker, n.published_at, n.headline, n.sentiment, n.relevance FROM news_items n "
            "JOIN stocks s ON s.id = n.stock_id"), ()
    if since is not None:
        q, p = q + " WHERE n.published_at >= ?", (pd.Timestamp(since).strftime("%Y-%m-%d"),)
    df = _obj_ticker(repo.db.query_df(q, p))
    if len(df):
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


# ------------------------------------------------------------------------------------------------ antrean EARNINGS
def earnings_queue(repo, cfg: dict, today) -> list[str]:
    today = pd.Timestamp(today).normalize()
    st = repo.stocks(active_only=True)
    if st.empty:
        return []
    active = list(st.sort_values(["board", "ticker"], key=lambda s: s.fillna("").astype(str)
                                 .str.replace("S&P 500", "0").str.replace("S&P 400", "1"))["ticker"])
    att = repo.db.query_df("SELECT symbol, MAX(called_at) AS last FROM api_usage WHERE provider = ? AND function = ? "
                           "AND status = 'OK' GROUP BY symbol", ("alpha_vantage", "EARNINGS"))
    last = dict(zip(att["symbol"], pd.to_datetime(att["last"], errors="coerce", utc=True).dt.tz_localize(None))) \
        if len(att) else {}
    sur = load_surprises(repo)
    last_rep = sur.groupby("ticker")["reported_date"].max().to_dict() if len(sur) else {}
    fresh_days = int(get(cfg, "alpha_vantage.fresh_report_days", 10) or 10)
    lo = today - pd.Timedelta(days=fresh_days)

    recent: dict[str, pd.Timestamp] = {}
    try:
        ev = repo.db.query_df("SELECT s.ticker, e.reaction_date FROM earnings_events e JOIN stocks s ON s.id = e.stock_id "
                              "WHERE e.reaction_date >= ?", (lo.strftime("%Y-%m-%d"),), parse_dates=["reaction_date"])
        for t, d in zip(ev["ticker"], ev["reaction_date"]):
            recent[t] = max(recent.get(t, d), d)
    except Exception:
        pass
    try:
        cal = load_calendar(repo)
        c = cal[(cal["report_date"] >= lo) & (cal["report_date"] < today)]
        for t, d in zip(c["ticker"], c["report_date"]):
            recent[t] = max(recent.get(t, d), d)
    except Exception:
        pass
    fresh = [t for t, d in sorted(recent.items(), key=lambda kv: kv[1], reverse=True)
             if t in set(active) and (t not in last_rep or last_rep[t] < d - pd.Timedelta(days=3))
             and (t not in last or last[t] < d + pd.Timedelta(days=2))]   # AV kadang telat ±1 hari → coba ≤ 2 hari
    never = [t for t in active if t not in last and t not in fresh]
    refresh = pd.Timedelta(days=int(get(cfg, "alpha_vantage.earnings_refresh_days", 120) or 120))
    stale = sorted((t for t in active if t in last and today - last[t] > refresh and t not in fresh),
                   key=lambda t: last[t])
    return fresh + never + stale


# ------------------------------------------------------------------------------------------------ run
def update_alpha_vantage(cfg: dict, repo, today=None, mode: str = "daily", client: AlphaVantageClient | None = None,
                         max_calls: int | None = None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    av = client or AlphaVantageClient(cfg, repo)
    out = {"mode": mode, "configured": av.configured(), "calls": 0, "calendar_rows": 0, "macro_rows": 0,
           "news_rows": 0, "earnings_tickers": 0, "earnings_rows": 0, "errors": [], "stopped": None}
    if not av.configured():
        repo.set_source_availability(SOURCE, "alt_data", "NOT_CONFIGURED",
                                     "GitHub Secret ALPHAVANTAGE_API_KEY belum diset", 3)
        return out
    out["budget_before"] = av.remaining()
    cap = max_calls if max_calls is not None else out["budget_before"]
    start_calls = av.calls

    def room() -> bool:
        return av.calls - start_calls < cap and av.remaining() > 0

    def step(name, fn):
        if not room():
            return None
        try:
            return fn()
        except QuotaExhausted as e:
            out["stopped"] = f"kuota habis: {e}"
            raise
        except PremiumOnly as e:
            out["errors"].append(f"{name}: khusus premium ({e})"[:200])
        except (AlphaVantageError, ValueError, KeyError) as e:
            out["errors"].append(f"{name}: {e}"[:200])
        return None

    try:
        if mode == "daily":
            every = int(get(cfg, "alpha_vantage.calendar_every_days", 1) or 1)
            last_cal = repo.db.scalar("SELECT MAX(called_at) FROM api_usage WHERE provider = ? AND function = ? AND "
                                      "status = 'OK'", ("alpha_vantage", "EARNINGS_CALENDAR"))
            if not last_cal or pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_cal) > pd.Timedelta(days=every, hours=-4):
                cal = step("EARNINGS_CALENDAR", av.earnings_calendar)
                if cal is not None:
                    out["calendar_rows"] = store_calendar(repo, cal, today)
                    out["calendar_universe"] = int(cal["ticker"].isin(set(repo.stock_ids())).sum())
            for mat in get(cfg, "alpha_vantage.treasury_maturities", ["10year"]) or []:
                d, _ = latest_series(repo, f"UST{str(mat).upper()}")
                if d is None or (today - d).days > int(get(cfg, "alpha_vantage.treasury_refresh_days", 2) or 2):
                    s = step(f"TREASURY_YIELD {mat}", lambda m=mat: av.treasury(m))
                    out["macro_rows"] += store_series(repo, s)
            d, _ = latest_series(repo, "FEDFUNDS")
            if d is None or (today - d).days > int(get(cfg, "alpha_vantage.fedfunds_refresh_days", 30) or 30):
                out["macro_rows"] += store_series(repo, step("FEDERAL_FUNDS_RATE", av.fed_funds))
            n = int(get(cfg, "alpha_vantage.news_calls_per_run", 2) or 0)
            end = pd.Timestamp.now(tz="America/New_York").tz_localize(None).floor("min")
            for k in range(n):
                hi = end - pd.Timedelta(hours=24 * k / n)
                lo = end - pd.Timedelta(hours=24 * (k + 1) / n)
                out["news_rows"] += store_news(repo, step("NEWS_SENTIMENT", lambda a=lo, b=hi: av.news(a, b)))
        for t in earnings_queue(repo, cfg, today):
            if not room():
                break
            df = step(f"EARNINGS {t}", lambda x=t: av.earnings(x))
            if df is not None:
                out["earnings_tickers"] += 1
                out["earnings_rows"] += store_surprises(repo, df)
                out["earnings_with_estimate"] = out.get("earnings_with_estimate", 0) + int(df["estimated_eps"].notna().sum())
                out["earnings_report_time"] = out.get("earnings_report_time", 0) + int(df["report_time"].notna().sum())
    except QuotaExhausted:
        pass
    out["calls"] = av.calls - start_calls
    out["budget_after"] = av.remaining()
    status = "AVAILABLE" if out["calls"] and len(out["errors"]) < max(1, out["calls"]) else (
        "ERROR" if out["errors"] else "AVAILABLE")
    cov = repo.db.scalar("SELECT COUNT(DISTINCT stock_id) FROM earnings_surprises") or 0
    out["surprise_coverage"] = int(cov)
    detail = (f"{out['calls']} request ({mode}); surprise EPS {cov} saham; kalender {out['calendar_rows']} baris baru; "
              f"berita {out['news_rows']}; sisa kuota {out['budget_after']}"
              + (f"; error: {out['errors'][0]}" if out["errors"] else ""))
    repo.set_source_availability(SOURCE, "alt_data", status, detail[:500], 3)
    log.info("Alpha Vantage: %s", detail, extra={"persist": True})
    return out


def risk_free_from_treasury(cfg: dict, repo, today=None) -> tuple[float | None, str | None]:
    """Yield 10Y terbaru sebagai risk-free (desimal) bila cukup segar."""
    if not get(cfg, "alpha_vantage.risk_free_from_treasury", True):
        return None, None
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    try:
        d, v = latest_series(repo, "UST10YEAR", upto=today)
    except Exception:
        return None, None
    if d is None or v is None or np.isnan(v):
        return None, None
    if (today - d).days > int(get(cfg, "alpha_vantage.risk_free_max_age_days", 10) or 10):
        return None, None
    return round(v / 100.0, 4), f"Treasury 10Y {d.date()} (Alpha Vantage)"


def apply_market_rates(cfg: dict, repo, today=None) -> str | None:
    """Ganti valuation.risk_free_rate (in-memory) dengan yield 10Y terbaru. Mengembalikan keterangan sumber."""
    rf, src = risk_free_from_treasury(cfg, repo, today)
    if rf is None:
        return None
    cfg.setdefault("valuation", {})["risk_free_rate"] = rf
    cfg["valuation"]["_risk_free_source"] = src
    return src
