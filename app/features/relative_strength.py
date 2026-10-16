import pandas as pd


def add_relative_strength(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, sector_df: pd.DataFrame | None = None):
    out = stock_df.copy()
    bench = benchmark_df[["date", "close"]].rename(columns={"close": "benchmark_close"})
    out = out.merge(bench, on="date", how="left")
    out["rs_vs_ihsg_20d"] = (
        out["close"].pct_change(20) - out["benchmark_close"].pct_change(20)
    )
    if sector_df is not None:
        sec = sector_df[["date", "close"]].rename(columns={"close": "sector_close"})
        out = out.merge(sec, on="date", how="left")
        out["rs_vs_sector_20d"] = (
            out["close"].pct_change(20) - out["sector_close"].pct_change(20)
        )
    return out
