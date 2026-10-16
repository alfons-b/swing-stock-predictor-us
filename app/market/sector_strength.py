"""Sector analysis (G) dan stock relative strength (H). Cross-sectional per tanggal → hanya
memakai data pada tanggal yang sama atau sebelumnya."""
from __future__ import annotations

import pandas as pd


def compute_sector_strength(df: pd.DataFrame) -> pd.DataFrame:
    d = df[~df["is_suspended"]]
    g = d.groupby(["date", "sector"])
    s = pd.DataFrame({
        "sector_ret5": g["ret5"].median(), "sector_ret20": g["ret20"].median(), "sector_ret60": g["ret60"].median(),
        "sector_breadth": (d["close"] > d["sma50"]).groupby([d["date"], d["sector"]]).mean(),
        "sector_volume_mom": g["f_volume_mom"].median(), "sector_n": g["ticker"].count(),
    }).reset_index()
    s = s.merge(df[["date", "idx_ret20"]].drop_duplicates("date"), on="date", how="left")
    s["sector_rs20"] = s["sector_ret20"] - s["idx_ret20"]
    comps = ["sector_ret5", "sector_ret20", "sector_ret60", "sector_breadth", "sector_rs20"]
    ranks = s.groupby("date")[comps].rank(pct=True)
    s["sector_score"] = 100 * ranks.mean(axis=1)
    return s.drop(columns=["idx_ret20"])


def add_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["f_rs20"] = df["ret20"] - df["idx_ret20"]
    df["f_rs60"] = df["ret60"] - df["idx_ret60"]
    df["f_rs5"] = df["ret5"] - df["idx_ret5"]
    df["f_rs_sector20"] = df["ret20"] - df["sector_ret20"]
    elig = ~df["is_suspended"]
    for col in ("f_rs20", "f_rs60"):
        df[col + "_rank"] = df[col].where(elig).groupby(df["date"]).rank(pct=True)
    return df
