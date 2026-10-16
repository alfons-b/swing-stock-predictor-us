import pandas as pd


def market_regime(df: pd.DataFrame) -> str:
    row = df.iloc[-1]
    close = row["close"]
    sma50, sma200 = row.get("sma50"), row.get("sma200")
    if pd.isna(sma50) or pd.isna(sma200):
        return "NEUTRAL"
    if close > sma50 > sma200 and row.get("rsi14", 50) >= 60:
        return "STRONG_BULL"
    if close > sma50 > sma200:
        return "BULL"
    if close < sma50 < sma200 and row.get("rsi14", 50) <= 40:
        return "STRONG_BEAR"
    if close < sma50 < sma200:
        return "BEAR"
    return "NEUTRAL"
