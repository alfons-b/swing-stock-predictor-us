"""Pembersihan & penyesuaian data. Setiap aksi dicatat di `log` untuk audit.

Keputusan desain:
- Baris rusak DIBUANG, bukan diinterpolasi (interpolasi = mengarang harga).
- Split/reverse split/rights disesuaikan mundur (backward adjustment) agar return historis
  benar. Ini menggunakan rasio corporate action yang diumumkan; ini praktik standar
  "adjusted price" dan tidak membocorkan arah harga masa depan.
- Kolom `value` (USD) TIDAK disesuaikan karena nilai transaksi tidak berubah oleh split.
- Baris suspensi tetap disimpan tapi diberi flag; fitur & label menghindarinya.
"""
from __future__ import annotations

import pandas as pd

from app.config import get
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def clean_prices(prices: pd.DataFrame, universe: pd.DataFrame, corporate_actions: pd.DataFrame, cfg: dict,
                 provider_adjusted: bool = False) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    df = prices.copy()
    n0 = len(df)

    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    notes.append(f"duplicate dibuang: {n0 - len(df)}")

    n1 = len(df)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"]) | (df["volume"] < 0)
    # high/low sedikit tidak konsisten terhadap open/close → perbaiki batasnya (bukan buang)
    df = df[~bad].copy()
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    notes.append(f"baris missing/invalid dibuang: {n1 - len(df)}")

    # --- batasi ke periode listing (point-in-time universe)
    u = universe.set_index("ticker")
    lst = df["ticker"].map(u["listing_date"])
    dls = df["ticker"].map(u["delisting_date"])
    keep = (lst.isna() | (df["date"] >= lst)) & (dls.isna() | (df["date"] <= dls))
    notes.append(f"baris di luar periode listing dibuang: {int((~keep).sum())}")
    df = df[keep]

    # --- corporate action adjustment
    df["adj_factor"] = 1.0
    if not provider_adjusted and len(corporate_actions):
        ca = corporate_actions.copy()
        for _, ev in ca.iterrows():
            action = str(ev["action"]).upper()
            m_t = df["ticker"] == ev["ticker"]
            m_before = m_t & (df["date"] < ev["ex_date"])
            if "is_adjusted" in df:  # baris dari provider yang sudah split-adjusted tidak disesuaikan dua kali
                m_before &= ~df["is_adjusted"].astype(bool)
            if not m_before.any():
                continue
            factor = None
            if action in ("SPLIT", "REVERSE_SPLIT", "BONUS") and pd.notna(ev.get("ratio")) and ev["ratio"] > 0:
                factor = 1.0 / float(ev["ratio"])
            elif action == "RIGHTS" and pd.notna(ev.get("ratio")) and ev["ratio"] > 0:
                factor = float(ev["ratio"])  # faktor penyesuaian harga dari TERP
            elif action == "DIVIDEND" and get(cfg, "quality.adjust_dividends", False) and pd.notna(ev.get("amount")):
                prev_close = df.loc[m_before, "close"].iloc[-1]
                factor = 1.0 - float(ev["amount"]) / prev_close
            if factor:
                df.loc[m_before, ["open", "high", "low", "close"]] *= factor
                df.loc[m_before, "volume"] /= factor
                df.loc[m_before, "adj_factor"] *= factor
                notes.append(f"{ev['ticker']} {action} {ev['ex_date'].date()}: harga sebelum ex-date x{factor:.4f}")

    # --- flags
    g = df.groupby("ticker", sort=False)
    zero = df["volume"] == 0
    run = zero.groupby([df["ticker"], (~zero).groupby(df["ticker"]).cumsum()]).cumsum()
    df["is_suspended"] = run >= get(cfg, "quality.suspension_min_zero_volume_days", 3)
    df["is_suspended"] |= zero  # hari tanpa transaksi tidak bisa dieksekusi
    ret = g["close"].pct_change()
    jump = ret.abs() > get(cfg, "quality.max_abs_daily_return", 0.35)
    known = set(zip(corporate_actions.get("ticker", []), pd.to_datetime(corporate_actions.get("ex_date", []))))
    df["ca_suspect"] = jump & ~pd.Series([(t, d) in known for t, d in zip(df["ticker"], df["date"])], index=df.index)
    # flag event corporate action terkini (risiko CA dalam 10 hari bursa terakhir)
    ca_day = pd.Series([(t, d) in known for t, d in zip(df["ticker"], df["date"])], index=df.index)
    df["ca_recent"] = (ca_day | df["ca_suspect"]).astype(int).groupby(df["ticker"]).transform(
        lambda s: s.rolling(10, min_periods=1).max()).astype(bool)
    df["days_listed"] = g.cumcount() + 1
    notes.append(f"flag suspended: {int(df['is_suspended'].sum())}, ca_suspect: {int(df['ca_suspect'].sum())}")
    for n in notes:
        log.info("cleaner: %s", n)
    return df.reset_index(drop=True), notes
