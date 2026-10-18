"""Fundamental (W) & news/sentiment (V) — POINT-IN-TIME.

- Fundamental di-join dengan merge_asof pada `available_date` (tanggal rilis publik),
  BUKAN `period_end`.
- Berita setelah jam penutupan (16:00 ET) baru dihitung pada hari bursa berikutnya.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FUND_COLS = ["revenue_growth", "eps_growth", "roe", "roa", "debt_to_equity", "net_margin", "pe"]
MARKET_CLOSE_HOUR = 16


def merge_fundamentals(df: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    if fund is None or fund.empty:
        return df
    f = fund.copy()
    f["available_date"] = pd.to_datetime(f["available_date"])
    cols = [c for c in FUND_COLS if c in f.columns]
    f = f[["ticker", "available_date"] + cols].sort_values("available_date")
    left = df.sort_values("date")
    out = pd.merge_asof(left, f.rename(columns={c: f"fund_{c}" for c in cols}), left_on="date",
                        right_on="available_date", by="ticker", direction="backward")
    out["fund_age_days"] = (out["date"] - out["available_date"]).dt.days
    return out.drop(columns=["available_date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def merge_news(df: pd.DataFrame, news: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if news is None or news.empty:
        return df
    n = news.copy()
    ts = pd.to_datetime(n["published_at"])
    eff = ts.dt.normalize()
    after = ts.dt.hour >= MARKET_CLOSE_HOUR
    eff = eff.where(~after, eff + pd.offsets.BDay(1))
    eff = eff.where(eff.dt.dayofweek < 5, eff + pd.offsets.BDay(1))
    n["date"] = eff.dt.normalize()
    s = n["sentiment"].astype(float)
    n["pos"], n["neg"] = (s > 0.2).astype(int), (s < -0.2).astype(int)
    daily = n.groupby(["ticker", "date"]).agg(news_count=("sentiment", "size"), news_sent=("sentiment", "mean"),
                                              news_pos=("pos", "sum"), news_neg=("neg", "sum")).reset_index()
    out = df.merge(daily, on=["ticker", "date"], how="left")
    for c in ("news_count", "news_pos", "news_neg"):
        out[c] = out[c].fillna(0)
    g = out.groupby("ticker", sort=False)
    out["news_count5"] = g["news_count"].transform(lambda x: x.rolling(window, min_periods=1).sum())
    out["news_pos5"] = g["news_pos"].transform(lambda x: x.rolling(window, min_periods=1).sum())
    out["news_neg5"] = g["news_neg"].transform(lambda x: x.rolling(window, min_periods=1).sum())
    wsum = (out["news_sent"].fillna(0) * out["news_count"])
    num = wsum.groupby(out["ticker"]).transform(lambda x: x.rolling(window, min_periods=1).sum())
    out["news_sent5"] = (num / out["news_count5"].replace(0, np.nan))
    base = g["news_count"].transform(lambda x: x.shift(window).rolling(60, min_periods=20).mean() * window)
    out["news_abnormal"] = out["news_count5"] / base.replace(0, np.nan)
    # Sebelum data berita mulai dikumpulkan, "0 berita" berarti TIDAK TAHU, bukan tidak ada berita → NaN.
    # Dengan begitu aturan cakupan fitur (> 50% baris training) menahan fitur berita dari model sampai riwayatnya
    # cukup panjang, alih-alih model belajar dari pola "dulu selalu 0".
    first = n["date"].min()
    before = out["date"] < first + pd.Timedelta(days=7)
    out.loc[before, ["news_count", "news_pos", "news_neg", "news_count5", "news_pos5", "news_neg5",
                     "news_abnormal"]] = np.nan
    return out


def fundamental_risk_flags(df: pd.DataFrame, cfg: dict) -> pd.Series:
    fc = cfg.get("fundamental_filter", {})
    if not fc.get("enabled", True) or "fund_debt_to_equity" not in df:
        return pd.Series(False, index=df.index)
    flag = pd.Series(False, index=df.index)
    # Bank & lembaga keuangan: utang (dana pihak ketiga) adalah bahan baku bisnis → D/E 5–8× normal.
    # Batas D/E hanya untuk sektor non-keuangan (sebelumnya semua bank akan tertolak).
    financial = pd.Series(False, index=df.index)
    if "sector" in df:
        from app.valuation.valuation_metrics import FINANCIAL_TYPES, sector_type
        sub = df["subsector"] if "subsector" in df else pd.Series(None, index=df.index)
        financial = pd.Series([sector_type(a, b) in FINANCIAL_TYPES for a, b in zip(df["sector"], sub)], index=df.index)
    flag |= (df["fund_debt_to_equity"] > fc.get("max_debt_to_equity", 4.0)) & ~financial
    flag |= df["fund_roe"] < fc.get("min_roe", -0.3)
    if fc.get("reject_negative_equity", True):
        flag |= df["fund_debt_to_equity"] < 0
    return flag.fillna(False)


def fundamental_score(df: pd.DataFrame) -> pd.Series:
    """0-100, cross-sectional per tanggal. NaN bila data tidak tersedia."""
    if "fund_roe" not in df:
        return pd.Series(np.nan, index=df.index)
    parts = {"fund_roe": 1, "fund_eps_growth": 1, "fund_revenue_growth": 1, "fund_net_margin": 1, "fund_debt_to_equity": -1}
    ranks = []
    for c, sign in parts.items():
        r = df[c].groupby(df["date"]).rank(pct=True)
        ranks.append(r if sign > 0 else 1 - r)
    return 100 * pd.concat(ranks, axis=1).mean(axis=1, skipna=True)


def sentiment_score(df: pd.DataFrame) -> pd.Series:
    if "news_sent5" not in df:
        return pd.Series(np.nan, index=df.index)
    s = 50 + 50 * df["news_sent5"].clip(-1, 1)
    return s.where(df["news_count5"] > 0)
