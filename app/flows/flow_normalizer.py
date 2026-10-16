"""Normalisasi data foreign flow mentah ke skema kanonik + pemeriksaan satuan.

Skema kanonik (satu baris = emiten × tanggal × segmen pasar):
  ticker, date, market_segment, foreign_buy_shares, foreign_sell_shares, net_foreign_shares,
  foreign_buy_value, foreign_sell_value, net_foreign_value, value_type, total_volume_shares, units, source,
  source_timestamp, quality_status, quality_notes

Aturan:
- LEMBAR dan NILAI disimpan terpisah. Nilai hanya diisi bila sumber memberikannya → value_type = ACTUAL_VALUE;
  bila tidak → value kosong + value_type = SHARES_ONLY. Estimasi (lembar × VWAP) dihitung di flow_features dan
  selalu berlabel ESTIMATED_VALUE — tidak pernah disimpan sebagai data mentah.
- `units`: data dalam LOT dikonversi ke lembar (× lot_size) dan dicatat.
- market_segment: REGULAR | NEGOTIATED | CASH | TOTAL (TOTAL = gabungan papan sebagaimana dilaporkan sumber).
- Pemeriksaan: beli/jual asing tidak boleh > volume total (indikasi salah satuan), tidak negatif, tidak duplikat.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.data.tickers import normalize_ticker

SEGMENTS = {"REGULAR", "NEGOTIATED", "CASH", "TOTAL"}
CANON = ["ticker", "date", "market_segment", "foreign_buy_shares", "foreign_sell_shares", "net_foreign_shares",
         "foreign_buy_value", "foreign_sell_value", "net_foreign_value", "value_type", "total_volume_shares", "units",
         "source", "source_timestamp", "quality_status", "quality_notes"]

ALIASES = {
    "ticker": ["ticker", "kode saham", "kode", "stock code", "code", "symbol"],
    "date": ["date", "tanggal", "trade date", "tanggal perdagangan"],
    "foreign_buy_shares": ["foreign_buy_shares", "foreign buy", "foreign buy volume", "fbuy", "beli asing", "asing beli"],
    "foreign_sell_shares": ["foreign_sell_shares", "foreign sell", "foreign sell volume", "fsell", "jual asing", "asing jual"],
    "foreign_buy_value": ["foreign_buy_value", "foreign buy value", "nilai beli asing"],
    "foreign_sell_value": ["foreign_sell_value", "foreign sell value", "nilai jual asing"],
    "total_volume_shares": ["total_volume_shares", "volume"],
    "market_segment": ["market_segment", "segment", "board", "papan"],
    "units": ["units", "satuan"],
}


def _norm(c) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower().replace("_", " "))


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {_norm(n): t for t, names in ALIASES.items() for n in names}
    mapping = {}
    for c in df.columns:
        t = lookup.get(_norm(c))
        if t and t not in mapping.values():
            mapping[c] = t
    return df.rename(columns=mapping)


def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True).replace("", np.nan)
    return pd.to_numeric(s, errors="coerce")


def normalize(df: pd.DataFrame, source: str, lot_size: int = 100, default_segment: str = "TOTAL",
              default_units: str = "shares") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CANON)
    d = rename_columns(df.copy())
    missing = {"ticker", "date", "foreign_buy_shares", "foreign_sell_shares"} - set(d.columns)
    if missing:
        raise ValueError(f"data foreign flow tanpa kolom wajib: {sorted(missing)}")
    out = pd.DataFrame()
    out["ticker"] = d["ticker"].map(lambda t: normalize_ticker(t) if pd.notna(t) and str(t).strip() else None)
    out["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    units = (d["units"].astype(str).str.lower().str.strip() if "units" in d else pd.Series(default_units, index=d.index))
    mult = np.where(units.str.startswith("lot"), lot_size, 1.0)
    for c in ("foreign_buy_shares", "foreign_sell_shares", "total_volume_shares"):
        out[c] = _to_num(d[c]) * mult if c in d else np.nan
    for c in ("foreign_buy_value", "foreign_sell_value"):
        out[c] = _to_num(d[c]) if c in d else np.nan
    out["net_foreign_shares"] = out["foreign_buy_shares"] - out["foreign_sell_shares"]
    out["net_foreign_value"] = out["foreign_buy_value"] - out["foreign_sell_value"]
    out["value_type"] = np.where(out["foreign_buy_value"].notna() & out["foreign_sell_value"].notna(), "ACTUAL_VALUE",
                                 "SHARES_ONLY")
    seg = d["market_segment"].astype(str).str.upper().str.strip() if "market_segment" in d else default_segment
    out["market_segment"] = seg
    out.loc[~out["market_segment"].isin(SEGMENTS), "market_segment"] = default_segment
    out["units"] = np.where(mult == 1.0, "shares", f"shares (from lots x{lot_size})")
    out["source"] = source
    out["source_timestamp"] = None
    out = out.dropna(subset=["ticker", "date"])
    return quality(out)


def quality(df: pd.DataFrame) -> pd.DataFrame:
    """Status per baris: OK | CHECK (indikasi satuan salah / negatif) | NO_TRADE (volume 0)."""
    notes = pd.Series("", index=df.index, dtype=object)
    status = pd.Series("OK", index=df.index, dtype=object)
    neg = (df["foreign_buy_shares"] < 0) | (df["foreign_sell_shares"] < 0)
    notes[neg] += "nilai negatif; "
    vol = df["total_volume_shares"]
    over = vol.notna() & ((df["foreign_buy_shares"] > vol * 1.0001) | (df["foreign_sell_shares"] > vol * 1.0001))
    notes[over] += "beli/jual asing > volume total — cek satuan (lot vs lembar) / segmen; "
    status[neg | over] = "CHECK"
    no_trade = vol.notna() & (vol == 0)
    status[no_trade & ~(neg | over)] = "NO_TRADE"
    df = df.copy()
    df["quality_status"], df["quality_notes"] = status, notes.str.strip()
    dup = df.duplicated(["ticker", "date", "market_segment"], keep="last")
    if dup.any():
        df = df[~dup]
    return df[CANON]
