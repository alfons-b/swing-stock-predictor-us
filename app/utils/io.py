"""I/O DataFrame. Parquet bila pyarrow tersedia, fallback ke pickle (cepat & lossless)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - tergantung environment
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except Exception:  # pragma: no cover
    _HAS_PARQUET = False


def save_df(df: pd.DataFrame, path_no_ext: Path) -> Path:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_PARQUET:
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    else:
        p = path_no_ext.with_suffix(".pkl")
        df.to_pickle(p)
    return p


def load_df(path_no_ext: Path) -> pd.DataFrame:
    for ext, reader in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        p = path_no_ext.with_suffix(ext)
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"Tidak ada file {path_no_ext}.parquet/.pkl — jalankan langkah sebelumnya terlebih dahulu.")


def exists_df(path_no_ext: Path) -> bool:
    return path_no_ext.with_suffix(".parquet").exists() or path_no_ext.with_suffix(".pkl").exists()


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.strftime("%Y-%m-%d")
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _clean_nan(obj):
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    return obj


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean_nan(obj), fh, cls=_Enc, indent=2, ensure_ascii=False)
    return path


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
