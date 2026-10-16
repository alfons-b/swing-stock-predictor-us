"""Model challenger: LSTM, GRU, ARIMA-GARCH, TFT (docs/CHALLENGER_MODELS.md).

Prinsip:
- TERPISAH dari model aktif (app/models): tidak memengaruhi keputusan BUY/WAIT/AVOID, tabel & workflow sendiri.
- Perbandingan adil: label (y_ret_5d, y_class), fold test (make_folds + embargo), biaya, dan metrik sama dengan
  model aktif.
- Modul di paket ini yang TIDAK mengimpor torch/statsmodels/arch: data, dist, evaluate, config (dipakai dashboard
  dan test tanpa dependensi berat). torch/statsmodels/arch hanya diimpor di arima_garch, nets, trainer.
"""
from __future__ import annotations

from app.config import get

MODEL_NAMES = ("arima_garch", "lstm", "gru", "tft")
SEQUENCE_MODELS = ("lstm", "gru", "tft")
MODEL_LABELS = {"arima_garch": "ARIMA-GARCH", "lstm": "LSTM", "gru": "GRU", "tft": "TFT"}


def challenger_cfg(cfg: dict, model: str | None = None) -> dict:
    """Konfigurasi efektif untuk satu model: blok umum + override per model (mis. tft.seq_len)."""
    base = dict(get(cfg, "challengers", {}) or {})
    seq = dict(base.get("sequence", {}) or {})
    tr = dict(base.get("training", {}) or {})
    arch_cfg = dict(base.get(model, {}) or {}) if model else {}
    for k in list(arch_cfg):
        if k in seq:
            seq[k] = arch_cfg.pop(k)
        elif k in tr:
            tr[k] = arch_cfg.pop(k)
    return {"quantiles": list(base.get("quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])),
            "min_history_days": int(base.get("min_history_days", 300)),
            "target_clip": float(base.get("target_clip", 0.5)),
            "baseline_feature": base.get("baseline_feature", "f_roc10"),
            "predict_lookback_days": int(base.get("predict_lookback_days", 420)),
            "artifacts_keep": int(base.get("artifacts_keep", 2)),
            "sequence": seq, "training": tr, "arch": arch_cfg,
            "horizon": int(get(cfg, "labels.SWING_HORIZON", 5)),
            "bull_threshold": float(get(cfg, "labels.bull_threshold", 0.03)),
            "bear_threshold": float(get(cfg, "labels.bear_threshold", -0.03)),
            "seed": int(get(cfg, "project.seed", 42))}


def enabled_models(cfg: dict) -> list[str]:
    en = get(cfg, "challengers.enabled", list(MODEL_NAMES)) or []
    return [m for m in MODEL_NAMES if m in en]
