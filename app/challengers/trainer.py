"""Pelatihan & inferensi model sekuens (LSTM, GRU, TFT) — PyTorch CPU, dibatasi waktu.

- Batch dibentuk dengan indeks numpy (tanpa DataLoader): jendela [t−L+1 .. t] emiten yang sama → tanpa look-ahead.
- Target distandardisasi dengan mean/std dari baris TRAINING (per horizon untuk TFT).
- Early stopping pada fold validasi (pinball loss); jumlah epoch terbaik dipakai untuk refit (test & produksi).
- Artifact = state_dict + konfigurasi + skala (tensor & tipe dasar saja) → dimuat dengan torch.load(weights_only=True),
  tidak pernah unpickle objek arbitrer dari database.
"""
from __future__ import annotations

import copy
import io
import os
import time

import numpy as np
import torch

from app.challengers.data import future_dows, window_index
from app.challengers.nets import build_model, pinball_loss
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def _setup_threads():
    n = os.cpu_count() or 1
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)


class SequenceSpec:
    """Semua yang dibutuhkan untuk membangun ulang model & input secara identik saat inferensi."""

    def __init__(self, kind: str, features: list, sectors: list, seq_len: int, horizon: int, levels: list,
                 arch: dict, center, scale, y_mu, y_sd, clip_z: float):
        self.kind, self.features, self.sectors = kind, list(features), list(sectors)
        self.seq_len, self.horizon, self.levels = int(seq_len), int(horizon), [float(x) for x in levels]
        self.arch = {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, (int, np.integer)) else v)
                     for k, v in (arch or {}).items()}
        self.center, self.scale = np.asarray(center, np.float32), np.asarray(scale, np.float32)
        self.y_mu, self.y_sd = np.atleast_1d(np.asarray(y_mu, np.float32)), np.atleast_1d(np.asarray(y_sd, np.float32))
        self.clip_z = float(clip_z)

    @property
    def median_index(self) -> int:
        return int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))

    def new_model(self) -> torch.nn.Module:
        return build_model(self.kind, len(self.features), len(self.sectors), self.horizon, self.arch,
                           len(self.levels), self.median_index)


def targets_for(kind: str, panel, ends: np.ndarray) -> np.ndarray:
    return panel.y_multi[ends] if kind == "tft" else panel.y[ends]


def finite_target_ends(kind: str, panel, ends: np.ndarray) -> np.ndarray:
    t = targets_for(kind, panel, ends)
    ok = np.isfinite(t).all(axis=1) if t.ndim == 2 else np.isfinite(t)
    return ends[ok]


def _batch(spec: SequenceSpec, panel, Z: np.ndarray, ends: np.ndarray):
    idx = window_index(ends, spec.seq_len)
    x = torch.from_numpy(Z[idx])
    static = torch.from_numpy(np.asarray(panel.sector[ends], np.int64))
    past_dow = torch.from_numpy(np.asarray(panel.dow[idx], np.int64))
    future_dow = torch.from_numpy(future_dows(panel.dates[ends], spec.horizon))
    return x, static, past_dow, future_dow


def _forward(model, spec: SequenceSpec, batch):
    x, static, past_dow, future_dow = batch
    return model(x, static, past_dow, future_dow)


def _std_targets(spec: SequenceSpec, panel, ends):
    t = targets_for(spec.kind, panel, ends)
    t = (t - spec.y_mu) / spec.y_sd if t.ndim == 2 else (t - spec.y_mu[0]) / spec.y_sd[0]
    return torch.from_numpy(np.asarray(t, np.float32))


def _loss(model, spec, panel, Z, ends, levels):
    return pinball_loss(_forward(model, spec, _batch(spec, panel, Z, ends)), _std_targets(spec, panel, ends), levels)


def fit_sequence_model(spec: SequenceSpec, panel, Z: np.ndarray, train_ends: np.ndarray, tr_cfg: dict, seed: int,
                       val_ends: np.ndarray | None = None, epochs: int | None = None) -> tuple[torch.nn.Module, dict]:
    """Latih satu model. Dengan `val_ends`: early stopping (bobot terbaik dipulihkan). Tanpa: tepat `epochs` epoch."""
    _setup_threads()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = spec.new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr_cfg.get("learning_rate", 1e-3)),
                            weight_decay=float(tr_cfg.get("weight_decay", 1e-4)))
    levels = torch.tensor(spec.levels, dtype=torch.float32)
    bs = int(tr_cfg.get("batch_size", 512))
    spe = int(tr_cfg.get("samples_per_epoch", 120000))
    max_epochs = int(epochs or tr_cfg.get("max_epochs", 12))
    min_epochs, patience = int(tr_cfg.get("min_epochs", 2)), int(tr_cfg.get("patience", 3))
    budget = float(tr_cfg.get("max_train_minutes", 20)) * 60
    clip = float(tr_cfg.get("grad_clip", 1.0))
    if val_ends is not None and len(val_ends) > int(tr_cfg.get("val_samples", 40000)):
        val_ends = np.sort(rng.choice(val_ends, int(tr_cfg.get("val_samples", 40000)), replace=False))
    t0, history, best = time.time(), [], (float("inf"), None, 0)
    bad, stopped = 0, "max_epochs"
    for ep in range(1, max_epochs + 1):
        model.train()
        sel = rng.choice(train_ends, size=min(spe, len(train_ends)), replace=False)
        losses = []
        for i in range(0, len(sel), bs):
            b = np.sort(sel[i:i + bs])
            loss = _loss(model, spec, panel, Z, b, levels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(float(loss))
            if time.time() - t0 > budget:
                stopped = "time_budget"
                break
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if val_ends is not None:
            row["val_loss"] = predict_loss(model, spec, panel, Z, val_ends, bs)
            if row["val_loss"] < best[0] - 1e-6:
                best, bad = (row["val_loss"], copy.deepcopy(model.state_dict()), ep), 0
            else:
                bad += 1
        history.append(row)
        log.info("[%s] epoch %d train %.5f%s (%.0fs)", spec.kind, ep, row["train_loss"],
                 f" val {row['val_loss']:.5f}" if "val_loss" in row else "", row["seconds"])
        if stopped == "time_budget":
            break
        if val_ends is not None and ep >= min_epochs and bad >= patience:
            stopped = "early_stopping"
            break
    if val_ends is not None and best[1] is not None:
        model.load_state_dict(best[1])
        best_epoch = best[2]
    else:
        best_epoch = len(history)
    model.eval()
    return model, {"history": history, "best_epoch": int(best_epoch), "epochs_run": len(history), "stopped": stopped,
                   "seconds": round(time.time() - t0, 1), "n_train_windows": int(len(train_ends))}


@torch.no_grad()
def predict_loss(model, spec, panel, Z, ends, bs: int = 1024) -> float:
    model.eval()
    levels = torch.tensor(spec.levels, dtype=torch.float32)
    tot, n = 0.0, 0
    for i in range(0, len(ends), bs):
        b = ends[i:i + bs]
        tot += float(_loss(model, spec, panel, Z, b, levels)) * len(b)
        n += len(b)
    model.train()
    return tot / max(1, n)


@torch.no_grad()
def predict_quantiles(model, spec: SequenceSpec, panel, Z: np.ndarray, ends: np.ndarray, bs: int = 1024) -> np.ndarray:
    """(N, Q) kuantil LOG return H hari (satuan asli) untuk baris `ends`."""
    model.eval()
    out = []
    for i in range(0, len(ends), bs):
        q = _forward(model, spec, _batch(spec, panel, Z, ends[i:i + bs])).numpy()
        if q.ndim == 3:                                                # TFT: horizon terakhir = H hari
            q = q[:, -1, :] * spec.y_sd[-1] + spec.y_mu[-1]
        else:
            q = q * spec.y_sd[0] + spec.y_mu[0]
        out.append(q)
    return np.concatenate(out) if out else np.zeros((0, len(spec.levels)))


# ------------------------------------------------------------------------------------------------ artifact
def save_artifact(model: torch.nn.Module, spec: SequenceSpec) -> bytes:
    buf = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "kind": spec.kind, "features": spec.features, "sectors": spec.sectors,
                "seq_len": spec.seq_len, "horizon": spec.horizon, "levels": spec.levels, "arch": spec.arch,
                "center": torch.from_numpy(spec.center), "scale": torch.from_numpy(spec.scale),
                "y_mu": torch.from_numpy(spec.y_mu), "y_sd": torch.from_numpy(spec.y_sd), "clip_z": spec.clip_z}, buf)
    return buf.getvalue()


def load_artifact(blob: bytes) -> tuple[torch.nn.Module, SequenceSpec]:
    d = torch.load(io.BytesIO(bytes(blob)), map_location="cpu", weights_only=True)
    spec = SequenceSpec(d["kind"], d["features"], d["sectors"], d["seq_len"], d["horizon"], d["levels"], d["arch"],
                        d["center"].numpy(), d["scale"].numpy(), d["y_mu"].numpy(), d["y_sd"].numpy(), d["clip_z"])
    model = spec.new_model()
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model, spec
