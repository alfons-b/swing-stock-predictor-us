"""Panel data untuk model challenger (numpy saja, tanpa torch).

Sumber: `build_dataset` yang SAMA dengan model aktif (fitur f_* bebas look-ahead, label y_* dari app/models/labels.py).
Baris diurutkan per emiten lalu tanggal; jendela sekuens untuk baris i = baris [i-L+1 .. i] emiten yang sama,
sehingga input pada tanggal t hanya memakai data ≤ t. Target = log(1 + y_ret_{H}d) (masa depan, hanya untuk label).

Split memakai `make_folds` (fold validasi terakhir untuk early stopping, fold test untuk evaluasi tersegel) dengan
embargo yang sama dengan model aktif.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.walk_forward import make_folds
from app.features.builder import feature_columns

ENGINEERED = ["c_ret1", "c_ret5", "c_dvol"]       # dinamika mentah untuk model sekuens (dihitung dari harga/volume)


@dataclass
class Panel:
    tickers: np.ndarray            # (N,) str
    dates: np.ndarray              # (N,) datetime64[ns]
    stock: np.ndarray              # (N,) int — indeks emiten
    pos: np.ndarray                # (N,) int — posisi baris di dalam histori emitennya (0 = baris pertama)
    X: np.ndarray                  # (N, F) float32 — fitur MENTAH (belum diskalakan)
    y: np.ndarray                  # (N,) float32 — log return H hari (NaN bila belum diketahui)
    y_multi: np.ndarray            # (N, H) float32 — log return kumulatif 1..H hari (target multi-horizon TFT)
    y_ret: np.ndarray              # (N,) float — return sederhana H hari (label asli)
    y_class: np.ndarray            # (N,) float — 0/1/2 atau NaN
    tradeable: np.ndarray          # (N,) bool
    sector: np.ndarray             # (N,) int — kode sektor (untuk input statis TFT)
    dow: np.ndarray                # (N,) int — hari dalam minggu (0=Senin)
    close: np.ndarray              # (N,) float
    features: list = field(default_factory=list)
    sectors: list = field(default_factory=list)

    def __len__(self):
        return len(self.dates)


def select_sequence_features(df: pd.DataFrame, train_mask: np.ndarray, max_features: int = 64,
                             min_coverage: float = 0.6, cfg: dict | None = None) -> list[str]:
    """Fitur f_* dengan cakupan tertinggi di baris TRAINING saja (deterministik, tanpa melihat target)."""
    feats = feature_columns(df, cfg)
    sub = df.loc[train_mask, feats]
    cov = sub.notna().mean()
    cov = cov[cov >= min_coverage]
    ranked = sorted(cov.index, key=lambda c: (-round(float(cov[c]), 6), c))
    return sorted(ranked[:max_features]) + ENGINEERED


def prepare_panel(df: pd.DataFrame, features: list[str], horizon: int, target_clip: float = 0.5,
                  sectors: list[str] | None = None) -> Panel:
    d = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = d.groupby("ticker", sort=False)
    close = d["close"].astype(float)
    d["c_ret1"] = np.log(close / g["close"].shift(1))
    d["c_ret5"] = np.log(close / g["close"].shift(5))
    vol = d["volume"].astype(float) if "volume" in d else pd.Series(np.nan, index=d.index)
    d["c_dvol"] = np.log1p(vol) - np.log1p(vol).groupby(d["ticker"]).shift(1)
    for c in features:
        if c not in d:
            d[c] = np.nan
    X = d[features].to_numpy(dtype=np.float32, na_value=np.nan)
    ycol = f"y_ret_{horizon}d"
    y_ret = d[ycol].to_numpy(dtype=float, na_value=np.nan) if ycol in d else np.full(len(d), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log1p(y_ret)
    y = np.clip(y, -target_clip, target_clip).astype(np.float32)
    y_multi = np.column_stack([np.log(g["close"].shift(-k) / close).to_numpy(dtype=float) for k in range(1, horizon + 1)])
    y_multi = np.clip(y_multi, -target_clip, target_clip).astype(np.float32)
    codes, uniq = pd.factorize(d["ticker"], sort=False)
    pos = d.groupby("ticker", sort=False).cumcount().to_numpy()
    sec_names = sectors if sectors is not None else sorted(d.get("sector", pd.Series(["Unknown"])).fillna("Unknown").unique())
    sec_map = {s: i + 1 for i, s in enumerate(sec_names)}              # 0 = sektor tidak dikenal saat inferensi
    sector = d.get("sector", pd.Series("Unknown", index=d.index)).fillna("Unknown").map(sec_map).fillna(0).astype(int)
    dates = pd.to_datetime(d["date"]).to_numpy()
    return Panel(tickers=d["ticker"].to_numpy(), dates=dates, stock=codes, pos=pos, X=X, y=y, y_multi=y_multi, y_ret=y_ret,
                 y_class=d["y_class"].to_numpy(dtype=float, na_value=np.nan) if "y_class" in d else np.full(len(d), np.nan),
                 tradeable=d["is_tradeable"].fillna(False).to_numpy(dtype=bool) if "is_tradeable" in d else np.ones(len(d), bool),
                 sector=sector.to_numpy(), dow=pd.to_datetime(d["date"]).dt.dayofweek.to_numpy(),
                 close=close.to_numpy(), features=list(features), sectors=list(sec_names))


# ------------------------------------------------------------------------------------------------ scaler
def fit_scaler(X: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median & IQR per fitur dari baris TRAINING saja (robust terhadap ekor tebal)."""
    sub = X[rows]
    if len(sub) > 300_000:                                             # cukup untuk statistik, hemat memori
        sub = sub[np.random.default_rng(0).choice(len(sub), 300_000, replace=False)]
    with np.errstate(all="ignore"):
        center = np.nanmedian(sub, axis=0)
        q75, q25 = np.nanpercentile(sub, 75, axis=0), np.nanpercentile(sub, 25, axis=0)
    scale = (q75 - q25) / 1.349
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def apply_scaler(X: np.ndarray, center: np.ndarray, scale: np.ndarray, clip: float = 5.0) -> np.ndarray:
    Z = (X - center) / scale
    Z = np.clip(Z, -clip, clip)
    Z[~np.isfinite(Z)] = 0.0                                           # NaN → median (0 setelah skala)
    return Z.astype(np.float32)


# ------------------------------------------------------------------------------------------------ jendela
def window_index(ends: np.ndarray, seq_len: int) -> np.ndarray:
    """(B, L) indeks baris untuk jendela yang berakhir di `ends` (baris t; hanya masa lalu & t)."""
    return np.asarray(ends)[:, None] + np.arange(-seq_len + 1, 1)[None, :]


def eligible_ends(panel: Panel, seq_len: int, date_lo=None, date_hi=None, need_target: bool = True,
                  tradeable_only: bool = True, min_history: int = 0) -> np.ndarray:
    m = panel.pos >= max(seq_len - 1, min_history - 1)
    if date_lo is not None:
        m &= panel.dates >= np.datetime64(pd.Timestamp(date_lo))
    if date_hi is not None:
        m &= panel.dates <= np.datetime64(pd.Timestamp(date_hi))
    if need_target:
        m &= np.isfinite(panel.y)
    if tradeable_only:
        m &= panel.tradeable
    return np.flatnonzero(m)


def latest_ends(panel: Panel, seq_len: int, min_history: int = 0) -> np.ndarray:
    """Baris terakhir tiap emiten yang punya data pada tanggal terbaru panel (untuk prakiraan harian)."""
    last = panel.dates.max()
    m = (panel.dates == last) & (panel.pos >= max(seq_len - 1, min_history - 1))
    return np.flatnonzero(m)


def future_dows(dates: np.ndarray, horizon: int) -> np.ndarray:
    """Hari dalam minggu untuk hari kerja t+1..t+H (input 'known future' TFT; libur bursa tidak diketahui → hari kerja)."""
    d = np.asarray(dates).astype("datetime64[D]")
    out = np.empty((len(d), horizon), dtype=np.int64)
    for k in range(1, horizon + 1):
        nxt = np.busday_offset(d, k, roll="forward")
        out[:, k - 1] = (nxt.astype("datetime64[D]").view("int64") - 4) % 7   # 1970-01-01 = Kamis → Senin = 0
    return out


# ------------------------------------------------------------------------------------------------ split
@dataclass
class SplitPlan:
    select_train_end: pd.Timestamp      # data training untuk seleksi epoch (≤ fold validasi terakhir − embargo)
    select_eval: tuple                  # (start, end) fold validasi terakhir → early stopping
    test_train_end: pd.Timestamp        # data training untuk evaluasi tersegel (≤ test − embargo)
    test_eval: tuple                    # (start, end) fold test (sama dengan test_sealed model aktif)
    train_start: pd.Timestamp


def split_plan(dates: pd.Series, cfg: dict) -> SplitPlan:
    folds = make_folds(dates, cfg)
    vals = [f for f in folds if f.kind == "validation"]
    tests = [f for f in folds if f.kind == "test"]
    if not vals or not tests:
        raise ValueError("Histori tidak cukup untuk fold validasi + test (lihat split.* di config/model.yaml)")
    v, t = vals[-1], tests[0]
    return SplitPlan(v.train_end, (v.eval_start, v.eval_end), t.train_end, (t.eval_start, t.eval_end), v.train_start)
