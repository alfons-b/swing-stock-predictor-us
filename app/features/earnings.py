"""Fitur & risiko earnings dari tanggal rilis laba resmi (SEC 8-K Item 2.02) — point-in-time.

Mengapa (audit 16 Okt 2026): 52% trade test keluar karena STOP, dan swing 5–10 hari yang menyeberangi earnings
menanggung gap ±5–15% yang melompati stop loss (eksekusi di open, kerugian > 1R). Di sisi lain, reaksi harga saat
earnings adalah salah satu anomali paling kokoh di AS: post-earnings announcement drift (Bernard & Thomas 1989);
versi tanpa data konsensus memakai abnormal return di sekitar pengumuman (EAR, Brandt dkk. 2008; Chan,
Jegadeesh & Lakonishok 1996).

Input `events`: ticker, reaction_date (hari bursa pertama harga bisa bereaksi — dihitung dari acceptanceDateTime
SEC: ≥ 16:00 ET → hari bursa berikutnya). Keluaran per (ticker, date):

  f_days_since_earn  hari bursa sejak reaksi earnings terakhir (0 = hari reaksi), dibatasi 126; NaN tanpa data
  f_ear3             abnormal return (saham − pasar) hari reaksi −1..+1; diketahui mulai reaksi+1, berlaku 63 hari bursa
  f_days_to_earn_est perkiraan hari KALENDER ke earnings berikutnya = reaksi terakhir + median 4 jarak terakhir − tanggal
  earn_soon_days     sama dengan f_days_to_earn_est (dipakai filter NO-TRADE `earnings_within_holding`)

Semua hanya memakai event yang reaksinya ≤ tanggal baris (EAR: ≤ tanggal − 1 hari bursa). Tanpa event → NaN (bukan 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EAR_VALID_DAYS = 63
DAYS_SINCE_CAP = 126


def reaction_date(accepted: pd.Series, filing_date: pd.Series) -> pd.Series:
    """acceptanceDateTime SEC (jam New York) → tanggal reaksi. Setelah 16:00 → hari kerja berikutnya."""
    # SEC menulis jam New York dengan akhiran "Z" yang menyesatkan → buang zona, perlakukan sebagai jam lokal ET
    txt = pd.Series(accepted, dtype=object).astype(str).str.replace(r"(Z|[+-]\d\d:?\d\d)$", "", regex=True)
    acc = pd.to_datetime(txt.where(~txt.isin(["None", "nan", "NaT", ""])), errors="coerce", format="mixed")
    base = acc.dt.normalize().fillna(pd.to_datetime(filing_date, errors="coerce"))
    after_close = acc.dt.hour >= 16
    nxt = base + pd.offsets.BDay(1)
    d = base.where(~after_close.fillna(False), nxt)
    # tanggal akhir pekan (8-K dikirim Sabtu) → Senin
    return d.where(d.dt.dayofweek < 5, d + pd.offsets.BDay(1))


def earnings_features(df: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["f_days_since_earn", "f_ear3", "f_days_to_earn_est", "earn_soon_days"]
    if events is None or len(events) == 0:
        for c in cols:
            df[c] = np.nan
        return df
    ev = events[["ticker", "reaction_date"]].dropna().copy()
    ev["reaction_date"] = pd.to_datetime(ev["reaction_date"]).dt.normalize()
    ev = ev.drop_duplicates().sort_values(["ticker", "reaction_date"])
    by_t = {t: g["reaction_date"].to_numpy() for t, g in ev.groupby("ticker", sort=False)}

    df = df.sort_values(["ticker", "date"]).copy()
    m1 = df[["date", "idx_close"]].drop_duplicates("date").set_index("date")["idx_close"].sort_index().pct_change()
    r1 = df.groupby("ticker", sort=False)["close"].pct_change()
    ar = (r1 - df["date"].map(m1)).to_numpy()
    since = np.full(len(df), np.nan)
    ear = np.full(len(df), np.nan)
    to_next = np.full(len(df), np.nan)
    dates_all = df["date"].to_numpy()
    tick_all = df["ticker"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tick_all[1:] != tick_all[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        rd = by_t.get(tick_all[s])
        if rd is None or len(rd) == 0:
            continue
        d = dates_all[s:e]
        # posisi hari reaksi di kalender saham ini (hari bursa pertama ≥ reaction_date)
        pos = np.searchsorted(d, rd, side="left")
        pos = pos[pos < len(d)]
        if len(pos) == 0:
            continue
        pos = np.unique(pos)
        a = ar[s:e]
        # indeks event terakhir yang reaksinya ≤ baris i
        last_ev = np.searchsorted(pos, np.arange(len(d)), side="right") - 1
        has = last_ev >= 0
        k = np.where(has, pos[np.clip(last_ev, 0, None)], 0)
        ds = np.where(has, np.arange(len(d)) - k, np.nan)
        since[s:e] = np.minimum(ds, DAYS_SINCE_CAP)
        # EAR3 per event (butuh hari reaksi+1 → baru diketahui di posisi k+1)
        ev_ear = np.array([np.nansum(a[max(0, p - 1):min(len(a), p + 2)]) if p + 1 < len(a) else np.nan for p in pos])
        idx = np.arange(len(d))
        last_known = np.searchsorted(pos + 1, idx, side="right") - 1      # event dengan k+1 ≤ i
        okk = last_known >= 0
        kk = np.where(okk, pos[np.clip(last_known, 0, None)], 0)
        valid = okk & (idx - kk <= EAR_VALID_DAYS)
        ear[s:e] = np.where(valid, ev_ear[np.clip(last_known, 0, None)], np.nan)
        # perkiraan earnings berikutnya dari jarak historis (hanya event ≤ hari ini)
        ev_dates = d[pos]
        gaps = np.diff(ev_dates).astype("timedelta64[D]").astype(float) if len(ev_dates) > 1 else np.array([])
        est = np.full(len(pos), np.datetime64("NaT"), dtype="datetime64[ns]")
        for j in range(len(pos)):                     # per EVENT (±24 untuk 6 tahun), bukan per baris
            g = gaps[max(0, j - 4):j]
            g = g[(g > 45) & (g < 140)]
            if len(g):
                est[j] = ev_dates[j] + np.timedelta64(int(np.median(g)), "D")
        rows = np.flatnonzero(has)
        e_i = est[last_ev[rows]]
        ok = ~np.isnat(e_i)
        days = (e_i[ok] - d[rows[ok]]).astype("timedelta64[D]").astype(float)
        to_next[s + rows[ok]] = np.maximum(0.0, days)
    df["f_days_since_earn"] = since
    df["f_ear3"] = ear
    df["f_days_to_earn_est"] = to_next
    df["earn_soon_days"] = to_next
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)
