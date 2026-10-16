"""Skema data standar internal. Setiap provider WAJIB mengembalikan kolom ini.

Keputusan desain: engine prediksi hanya mengenal skema ini, sehingga sumber data
(Yahoo, CSV, feed berlisensi) bisa diganti tanpa mengubah kode lain.
"""
from __future__ import annotations

import pandas as pd

UNIVERSE_COLUMNS = {
    "ticker": "str — identifier internal format Yahoo, mis. AAPL, BRK-B",
    "name": "str — nama emiten",
    "sector": "str — sektor GICS (atau taksonomi Yahoo)",
    "subsector": "str — subsektor",
    "listing_date": "date — tanggal pencatatan (deteksi IPO baru)",
    "delisting_date": "date atau null — wajib diisi untuk saham delisting (hindari survivorship bias)",
    "board": "str atau null — indeks/bursa asal (S&P 500, S&P 400, NYSE, Nasdaq, OTC)",
}

PRICE_COLUMNS = {
    "ticker": "str", "date": "datetime64 (tanggal perdagangan)",
    "open": "float", "high": "float", "low": "float", "close": "float",
    "volume": "float — saham", "value": "float — nilai transaksi USD (estimasi close*volume bila tidak tersedia)",
    "frequency": "float atau NaN — jumlah transaksi",
}

INDEX_COLUMNS = {"date": "datetime64", "open": "float", "high": "float", "low": "float", "close": "float", "volume": "float"}

CORPORATE_ACTION_COLUMNS = {
    "ticker": "str", "ex_date": "date",
    "action": "SPLIT | REVERSE_SPLIT | RIGHTS | DIVIDEND | BONUS",
    "ratio": "float — faktor jumlah saham baru/lama (split 1:5 → 5; reverse 5:1 → 0.2); untuk RIGHTS = faktor penyesuaian harga",
    "amount": "float — dividen per saham (USD)",
}

FUNDAMENTAL_COLUMNS = {
    "ticker": "str", "period_end": "date",
    "available_date": "date — tanggal laporan PUBLIK (point-in-time, bukan period_end!)",
    "revenue_growth": "float", "eps_growth": "float", "roe": "float", "roa": "float",
    "debt_to_equity": "float", "net_margin": "float", "pe": "float",
}

NEWS_COLUMNS = {"ticker": "str", "published_at": "datetime", "headline": "str", "sentiment": "float -1..1"}

PRICE_DTYPES = {"open": float, "high": float, "low": float, "close": float, "volume": float, "value": float, "frequency": float}


def conform_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if "value" not in df or df["value"].isna().all():
        df["value"] = df["close"] * df["volume"]
    if "frequency" not in df:
        df["frequency"] = float("nan")
    for c, t in PRICE_DTYPES.items():
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(t)
    df["ticker"] = df["ticker"].astype(str)
    return df[list(PRICE_COLUMNS)].sort_values(["ticker", "date"]).reset_index(drop=True)


def conform_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in UNIVERSE_COLUMNS:
        if c not in df:
            df[c] = None
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df["delisting_date"] = pd.to_datetime(df["delisting_date"], errors="coerce")
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    df["subsector"] = df["subsector"].fillna("Unknown").astype(str)
    df["name"] = df["name"].fillna(df["ticker"]).astype(str)
    return df[list(UNIVERSE_COLUMNS)].drop_duplicates("ticker", keep="last").reset_index(drop=True)
