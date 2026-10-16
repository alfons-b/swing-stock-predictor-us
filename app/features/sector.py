import pandas as pd


def sector_strength(stock_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sector, df in stock_frames.items():
        if len(df) < 61:
            continue
        close = df["close"]
        volume = df["volume"]
        rows.append({
            "sector": sector,
            "return_5d": close.pct_change(5).iloc[-1],
            "return_20d": close.pct_change(20).iloc[-1],
            "return_60d": close.pct_change(60).iloc[-1],
            "volume_momentum": volume.rolling(20).mean().pct_change(20).iloc[-1],
        })
    if not rows:
        return pd.DataFrame(columns=[
            "sector","return_5d","return_20d","return_60d","volume_momentum","strength"
        ])
    out = pd.DataFrame(rows)
    for col in ["return_5d","return_20d","return_60d","volume_momentum"]:
        out[f"{col}_rank"] = out[col].rank(pct=True)
    out["strength"] = (
        0.20 * out["return_5d_rank"] +
        0.35 * out["return_20d_rank"] +
        0.30 * out["return_60d_rank"] +
        0.15 * out["volume_momentum_rank"]
    ) * 100
    return out.sort_values("strength", ascending=False)
