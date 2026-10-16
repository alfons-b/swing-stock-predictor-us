"""Data quality checks (bagian C). Validator TIDAK mengubah data — hanya melaporkan.

Setiap check menghasilkan: nama, severity (INFO/WARN/ERROR), jumlah baris, contoh ticker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from app.config import get


@dataclass
class Issue:
    check: str
    severity: str
    count: int
    detail: str = ""
    tickers: list = field(default_factory=list)


def _issue(check, severity, mask_or_count, df=None, detail=""):
    if isinstance(mask_or_count, (int, np.integer)):
        return Issue(check, severity, int(mask_or_count), detail)
    n = int(mask_or_count.sum())
    tk = sorted(df.loc[mask_or_count, "ticker"].unique().tolist())[:10] if df is not None and n else []
    return Issue(check, severity if n else "OK", n, detail, tk)


def validate_prices(prices: pd.DataFrame, universe: pd.DataFrame, corporate_actions: pd.DataFrame,
                    cfg: dict, provider_adjusted: bool = False) -> list[Issue]:
    q = cfg.get("quality", {})
    df = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    issues: list[Issue] = []

    # 1. missing data
    issues.append(_issue("missing_ohlcv", "WARN", df[["open", "high", "low", "close", "volume"]].isna().any(axis=1), df))
    # 2. duplicate date
    issues.append(_issue("duplicate_date", "WARN", df.duplicated(["ticker", "date"], keep=False), df))
    # 3. harga tidak valid
    bad_px = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"]) | \
             (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    issues.append(_issue("invalid_price", "ERROR", bad_px, df, "harga <= 0 atau high/low tidak konsisten"))
    # 4. volume tidak valid
    issues.append(_issue("invalid_volume", "ERROR", df["volume"] < 0, df))
    # 5 & 14. suspensi / trading halt (volume 0 beruntun, harga beku)
    zero = (df["volume"] == 0)
    run = zero.groupby((~zero).cumsum()).cumsum()
    susp = run >= get(cfg, "quality.suspension_min_zero_volume_days", 3)
    issues.append(_issue("suspension_or_halt", "WARN", susp, df, "volume 0 beruntun"))
    # 6. IPO baru
    first = df.groupby("ticker")["date"].transform("min")
    hist = df.groupby("ticker").cumcount()
    issues.append(_issue("ipo_short_history", "INFO", (hist < q.get("min_history_days", 250)) & (first > df["date"].min()), df))
    # 7-11. corporate action
    ret = df.groupby("ticker")["close"].pct_change()
    jump = ret.abs() > q.get("max_abs_daily_return", 0.35)
    known = set(zip(corporate_actions.get("ticker", []), pd.to_datetime(corporate_actions.get("ex_date", []))))
    explained = pd.Series([(t, d) in known for t, d in zip(df["ticker"], df["date"])], index=df.index)
    issues.append(_issue("corporate_action_events", "INFO", len(corporate_actions), detail=
                         ", ".join(sorted(corporate_actions["action"].astype(str).unique())) if len(corporate_actions) else "tidak ada data CA"))
    issues.append(_issue("split_or_reverse_split_unadjusted", "WARN" if not provider_adjusted else "INFO",
                         jump & explained, df, "lonjakan harga di tanggal corporate action → perlu adjustment"))
    # 13. outlier abnormal (tanpa penjelasan corporate action)
    issues.append(_issue("unexplained_price_jump", "WARN", jump & ~explained, df,
                         "kemungkinan split tidak tercatat / error data — baris diberi flag ca_suspect"))
    # 12. perubahan struktur perdagangan: lonjakan median nilai transaksi 60h vs 250h sebelumnya
    med60 = df.groupby("ticker")["value"].transform(lambda s: s.rolling(60, min_periods=30).median())
    med250 = df.groupby("ticker")["value"].transform(lambda s: s.shift(60).rolling(250, min_periods=120).median())
    regime_shift = (med60 / med250).replace([np.inf, -np.inf], np.nan)
    issues.append(_issue("trading_structure_change", "INFO", (regime_shift > 10) | (regime_shift < 0.1), df,
                         "nilai transaksi berubah >10x — cek perubahan papan/free float"))
    # gap tanggal (data hilang beberapa hari padahal emiten aktif)
    gaps = df.groupby("ticker")["date"].diff().dt.days > 10
    issues.append(_issue("calendar_gap_gt_10d", "WARN", gaps, df))
    # 15. survivorship bias
    n_delisted = int(universe["delisting_date"].notna().sum())
    issues.append(Issue("survivorship_bias", "OK" if n_delisted else "WARN", n_delisted,
                        "jumlah emiten delisting di universe" if n_delisted else
                        "TIDAK ADA emiten delisting di universe → hasil backtest kemungkinan bias ke atas"))
    unknown = set(df["ticker"]) - set(universe["ticker"])
    issues.append(Issue("ticker_not_in_universe", "WARN" if unknown else "OK", len(unknown), "", sorted(unknown)[:10]))
    return issues


def issues_to_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([asdict(i) for i in issues])
