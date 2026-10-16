"""Walk-forward / expanding window dengan embargo (§30).

mode rolling (default, untuk retrain otomatis):
  test       = `test_months` terakhir (data terbaru yang tidak tersentuh tuning/seleksi)
  validation = `validation_folds` × `fold_months` tepat sebelum test
  train      = dari awal data (+warmup) s/d awal fold − embargo (expanding)
mode fixed: tahun validasi & TEST_START dari config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.config import embargo_days, get


TRADING_DAYS_PER_YEAR = 252  # hari bursa NYSE per tahun (Senin–Jumat dikurangi ±9–10 libur)


class SplitError(RuntimeError):
    """Histori data terlalu pendek untuk walk-forward yang diminta."""


def history_requirement(cfg: dict, n_folds: int | None = None) -> float:
    """Perkiraan histori minimum (tahun) untuk `n_folds` fold validasi (default: jumlah dari config)."""
    n = int(get(cfg, "split.validation_folds", 4)) if n_folds is None else n_folds
    warm = int(get(cfg, "split.warmup_trading_days", 260)) / TRADING_DAYS_PER_YEAR
    train = int(get(cfg, "split.min_train_months", 12)) / 12
    fold = int(get(cfg, "split.fold_months", 6)) / 12
    test = int(get(cfg, "split.test_months", 6)) / 12
    return round(warm + train + n * fold + test + 0.1, 1)  # +0.1 th margin embargo


def assert_feasible(dates, cfg: dict) -> list:
    """Pastikan minimal 1 fold validasi + fold test terbentuk; bila tidak, error yang menjelaskan solusinya."""
    if get(cfg, "split.mode", "rolling") != "rolling":
        folds = make_folds(dates, cfg)
        if not any(f.kind == "validation" for f in folds):
            raise SplitError("Tidak ada fold validasi — periksa split.VALIDATION_YEARS vs rentang data (mode fixed)")
        return folds
    ud = pd.to_datetime(pd.Series(dates).unique())
    folds = make_folds(dates, cfg) if len(ud) > int(get(cfg, "split.warmup_trading_days", 260)) else []
    n_val = sum(f.kind == "validation" for f in folds)
    if n_val == 0:
        span = (ud.max() - ud.min()).days / 365.25 if len(ud) else 0
        raise SplitError(
            f"Histori data terlalu pendek untuk walk-forward: {len(ud)} hari bursa (~{span:.1f} tahun). "
            f"Minimal ~{history_requirement(cfg, 1)} tahun untuk 1 fold validasi, ~{history_requirement(cfg)} tahun untuk "
            f"{get(cfg, 'split.validation_folds', 4)} fold. Jalankan ulang dengan histori lebih panjang, mis. "
            f"`--set data.initial_history_years=5` — data yang sudah ada tidak diunduh ulang, hanya tahun yang kurang.")
    return folds


@dataclass
class Fold:
    name: str
    kind: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    def as_dict(self):
        return {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v) for k, v in asdict(self).items()}


def _embargoed_end(ud: np.ndarray, eval_start, emb: int) -> pd.Timestamp:
    pos = np.searchsorted(ud, np.datetime64(pd.Timestamp(eval_start)))
    return pd.Timestamp(ud[max(0, pos - emb - 1)])


def resolve_periods(dates: pd.Series, cfg: dict):
    ud = np.sort(pd.to_datetime(pd.Series(dates).unique()).to_numpy())
    last = pd.Timestamp(ud[-1])
    if get(cfg, "split.mode", "rolling") == "fixed":
        ts = pd.Timestamp(get(cfg, "split.TRAIN_START"))
        test_start = pd.Timestamp(get(cfg, "split.TEST_START"))
        test_end = pd.Timestamp(get(cfg, "split.TEST_END")) if get(cfg, "split.TEST_END") else last
        vals = [(f"val_{y}", pd.Timestamp(f"{y}-01-01"), min(pd.Timestamp(f"{y}-12-31"), test_start - pd.Timedelta(days=1)))
                for y in get(cfg, "split.VALIDATION_YEARS", []) if pd.Timestamp(f"{y}-01-01") < test_start]
        return ts, vals, test_start, test_end
    warm = int(get(cfg, "split.warmup_trading_days", 260))
    ts = pd.Timestamp(ud[min(warm, len(ud) - 1)])
    test_start = last - pd.DateOffset(months=int(get(cfg, "split.test_months", 6))) + pd.Timedelta(days=1)
    fm, n = int(get(cfg, "split.fold_months", 6)), int(get(cfg, "split.validation_folds", 4))
    vals = []
    end = test_start - pd.Timedelta(days=1)
    for i in range(n):
        start = end - pd.DateOffset(months=fm) + pd.Timedelta(days=1)
        if start <= ts + pd.DateOffset(months=int(get(cfg, "split.min_train_months", 12))):  # training minimal
            break
        vals.append((f"val_{start:%Y%m}", start, end))
        end = start - pd.Timedelta(days=1)
    return ts, list(reversed(vals)), test_start, last


def make_folds(dates: pd.Series, cfg: dict, include_test: bool = True) -> list[Fold]:
    ud = np.sort(pd.to_datetime(pd.Series(dates).unique()).to_numpy())
    emb = embargo_days(cfg)
    ts, vals, test_start, test_end = resolve_periods(dates, cfg)
    folds = [Fold(n, "validation", ts, _embargoed_end(ud, s, emb), s, e) for n, s, e in vals]
    if include_test and test_start <= pd.Timestamp(ud[-1]):
        folds.append(Fold("test", "test", ts, _embargoed_end(ud, test_start, emb), test_start, test_end))
    for f in folds:
        gap = ((ud > np.datetime64(f.train_end)) & (ud < np.datetime64(f.eval_start))).sum()
        assert gap >= emb, f"Embargo dilanggar di {f.name}"
        assert f.train_end < f.eval_start
    return folds


def fold_masks(df: pd.DataFrame, fold: Fold):
    d = df["date"]
    return ((d >= fold.train_start) & (d <= fold.train_end)).to_numpy(), \
        ((d >= fold.eval_start) & (d <= fold.eval_end)).to_numpy()
