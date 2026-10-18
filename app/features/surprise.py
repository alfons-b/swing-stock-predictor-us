"""Kejutan laba (EPS aktual vs konsensus analis) dan kalender rilis yang pasti — dari Alpha Vantage, point-in-time.

Mengapa: f_ear3 (reaksi harga) hanya proksi kejutan. Kejutan EPS terhadap konsensus adalah sinyal post-earnings
announcement drift yang asli (Bernard & Thomas 1989; Livnat & Mendenhall 2006), dan menggabungkan keduanya
memperkuat drift (Chan, Jegadeesh & Lakonishok 1996; Brandt dkk. 2008). Pola "beat beruntun" juga persisten
(Loh & Warachka 2012).

Fitur (berlaku 91 hari kalender setelah diketahui, sesudahnya NaN):
  f_sue              surprise ÷ simpangan baku surprise 8 kuartal terakhir (min. 4) — standardized unexpected earnings
  f_eps_surprise_pct surprise ÷ |estimasi| (desimal, dipotong ±1)
  f_beat_streak      jumlah kuartal beruntun EPS > estimasi (0–8)

Kapan diketahui: rilis "pre-market" → hari itu; "post-market" atau jam tidak diketahui → hari bursa berikutnya
(konservatif: tidak pernah memakai angka sebelum publik).

Kalender (`apply_earnings_calendar`): untuk tanggal yang sudah punya snapshot EARNINGS_CALENDAR (≤ 7 hari),
earn_soon_days memakai tanggal rilis RESMI berikutnya, bukan perkiraan dari pola SEC. Hanya filter yang memakai ini —
fitur model f_days_to_earn_est tetap perkiraan agar training dan scan live konsisten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_CAL_DAYS = 91
SNAPSHOT_MAX_AGE_DAYS = 7
SURPRISE_COLS = ["f_sue", "f_eps_surprise_pct", "f_beat_streak"]


def effective_date(reported_date: pd.Series, report_time: pd.Series) -> pd.Series:
    d = pd.to_datetime(reported_date, errors="coerce").dt.normalize()
    rt = pd.Series(report_time, index=d.index).fillna("").astype(str).str.lower()
    pre = rt.str.startswith("pre")
    d = d.where(pre, d + pd.offsets.BDay(1))
    return d.where(d.dt.dayofweek < 5, d + pd.offsets.BDay(1))


def surprise_table(sur: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per (ticker, kuartal) dengan fitur yang sudah dihitung, urut tanggal diketahui."""
    s = sur.dropna(subset=["reported_date"]).copy()
    s["eff"] = effective_date(s["reported_date"], s.get("report_time", pd.Series(index=s.index, dtype=object)))
    s = s.sort_values(["ticker", "eff"]).drop_duplicates(["ticker", "eff"], keep="last")
    g = s.groupby("ticker", sort=False)["surprise"]
    sd = g.transform(lambda x: x.rolling(8, min_periods=4).std())
    s["f_sue"] = (s["surprise"] / sd.where(sd > 1e-9)).clip(-5, 5)
    est = s["estimated_eps"].abs()
    pct = s["surprise"] / est.where(est >= 0.01)
    if "surprise_pct" in s:
        pct = pct.fillna(s["surprise_pct"] / 100.0)
    s["f_eps_surprise_pct"] = pct.clip(-1, 1)
    beat = (s["surprise"] > 0).astype(float).where(s["surprise"].notna())

    def streak(b):
        out, run = [], 0
        for v in b:
            run = 0 if (v != 1) else min(run + 1, 8)
            out.append(run if v == v else np.nan)
        return pd.Series(out, index=b.index)
    s["f_beat_streak"] = beat.groupby(s["ticker"], sort=False).transform(streak)
    return s[["ticker", "eff"] + SURPRISE_COLS]


def add_surprise_features(df: pd.DataFrame, sur: pd.DataFrame | None) -> pd.DataFrame:
    if sur is None or len(sur) == 0:
        for c in SURPRISE_COLS:
            df[c] = np.nan
        return df
    tab = surprise_table(sur).sort_values("eff")
    tab["ticker"] = tab["ticker"].astype(object)            # merge_asof menolak object vs StringDtype
    left = df.sort_values("date").drop(columns=[c for c in SURPRISE_COLS if c in df.columns])
    left["ticker"] = left["ticker"].astype(object)
    out = pd.merge_asof(left, tab, left_on="date", right_on="eff", by="ticker", direction="backward")
    stale = (out["date"] - out["eff"]).dt.days > VALID_CAL_DAYS
    for c in SURPRISE_COLS:
        out.loc[stale, c] = np.nan
    return out.drop(columns=["eff"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def next_report_dates(cal: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    """Tanggal rilis berikutnya per ticker menurut kalender yang diketahui pada `day` (log perubahan → state)."""
    c = cal[cal["as_of"] <= day]
    if c.empty or (day - c["as_of"].max()).days > SNAPSHOT_MAX_AGE_DAYS:
        return pd.Series(dtype="datetime64[ns]")
    c = c.sort_values("as_of").drop_duplicates(["ticker", "fiscal_date_ending"], keep="last")
    c = c[c["report_date"] >= day]
    return c.groupby("ticker")["report_date"].min()


def apply_earnings_calendar(df: pd.DataFrame, cal: pd.DataFrame | None) -> pd.DataFrame:
    if "earn_soon_days" not in df:
        df["earn_soon_days"] = np.nan
    df["earn_date_source"] = np.where(df["earn_soon_days"].notna(), "estimate", None)
    if cal is None or len(cal) == 0:
        return df
    cal = cal.copy()
    for c in ("as_of", "report_date", "fiscal_date_ending"):
        cal[c] = pd.to_datetime(cal[c]).dt.normalize()
    first = cal["as_of"].min()
    for day in sorted(d for d in df["date"].unique() if pd.Timestamp(d) >= first):
        day = pd.Timestamp(day)
        nxt = next_report_dates(cal, day)
        if nxt.empty:
            continue
        m = (df["date"] == day) & df["ticker"].isin(nxt.index)
        if m.any():
            days = (df.loc[m, "ticker"].map(nxt) - day).dt.days
            df.loc[m, "earn_soon_days"] = days.to_numpy()
            df.loc[m, "earn_date_source"] = "calendar"
    return df
