"""Pemuatan konfigurasi: config/app.yaml + sources.yaml + model.yaml + trading.yaml (+ research.yaml) digabung jadi satu dict.

- `${VAR:-default}` disubstitusi dari environment SEBELUM parsing YAML (tipe angka/bool tetap benar).
- Secrets (DATABASE_URL, API key provider, ...) hanya dari environment / GitHub Secrets, tidak pernah dari file.
- Override CLI: `--set strategy.MIN_RISK_REWARD=2.0`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = ["app.yaml", "sources.yaml", "model.yaml", "trading.yaml"]
OPTIONAL_CONFIG_FILES = ["research.yaml", "challengers.yaml"]   # riset & model challenger (default aman bila tidak ada)
_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = ["project.seed", "database.url", "labels.SWING_HORIZON", "labels.HORIZONS", "split.mode",
                 "strategy.MIN_RISK_REWARD", "portfolio.RISK_PER_TRADE", "backtest.TRANSACTION_FEE_BUY",
                 "scoring.TOP_N_STOCKS", "calendar.timezone", "market_data.providers"]


class ConfigError(ValueError):
    pass


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_value(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _substitute_env(text: str) -> str:
    def rep(m):
        val = os.environ.get(m.group(1))
        return val if val not in (None, "") else (m.group(2) or "")
    return _ENV.sub(rep, text)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(config_dir: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cdir = Path(config_dir) if config_dir else ROOT / "config"
    if not cdir.is_absolute():
        cdir = (Path.cwd() / cdir) if (Path.cwd() / cdir).exists() else ROOT / cdir
    cfg: dict = {}
    for name in CONFIG_FILES + OPTIONAL_CONFIG_FILES:
        p = cdir / name
        if not p.exists():
            if name in OPTIONAL_CONFIG_FILES:
                continue
            raise ConfigError(f"File config tidak ditemukan: {p}")
        cfg = _deep_merge(cfg, yaml.safe_load(_substitute_env(p.read_text(encoding="utf-8"))) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override harus key=value: {item}")
        k, v = item.split("=", 1)
        set_value(cfg, k.strip(), yaml.safe_load(v))
    # trading.* adalah alias ramah-pengguna untuk parameter inti
    t = cfg.get("trading", {})
    if "minimum_rr" in t:
        cfg["strategy"]["MIN_RISK_REWARD"] = t["minimum_rr"]
    if "risk_per_trade" in t:
        cfg["portfolio"]["RISK_PER_TRADE"] = t["risk_per_trade"]
    if "horizon" in t:
        cfg["labels"]["SWING_HORIZON"] = t["horizon"]
    validate_config(cfg)
    cfg["_root"] = str(cdir.parent.resolve())
    cfg["_config_dir"] = str(cdir.resolve())
    return cfg


def validate_config(cfg: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if get(cfg, k) is None]
    if missing:
        raise ConfigError(f"Key config wajib tidak ada: {missing}")
    if get(cfg, "labels.SWING_HORIZON") not in get(cfg, "labels.HORIZONS"):
        raise ConfigError("labels.SWING_HORIZON (trading.horizon) harus ada di labels.HORIZONS")
    if get(cfg, "labels.bull_threshold") <= get(cfg, "labels.bear_threshold"):
        raise ConfigError("bull_threshold harus > bear_threshold")
    w = get(cfg, "scoring.technical_weights", {})
    if abs(sum(w.values()) - 1.0) > 1e-6:
        raise ConfigError(f"scoring.technical_weights harus berjumlah 1.0 (sekarang {sum(w.values()):.3f})")
    tp = get(cfg, "strategy.TP_MULTIPLIER")
    if not (isinstance(tp, list) and len(tp) == 2 and 0 < tp[0] < tp[1]):
        raise ConfigError("strategy.TP_MULTIPLIER harus [tp1, tp2] dengan 0 < tp1 < tp2")
    if get(cfg, "strategy.MIN_CONFIDENCE") is None and get(cfg, "strategy.MIN_PROB_EDGE") is None:
        raise ConfigError("Isi strategy.MIN_CONFIDENCE atau strategy.MIN_PROB_EDGE")
    if not 0 < float(get(cfg, "portfolio.RISK_PER_TRADE")) <= 0.05:
        raise ConfigError("RISK_PER_TRADE harus di (0, 0.05]")
    keep = get(cfg, "retention.price_history_years")
    if keep is not None and (float(keep) < float(get(cfg, "data.initial_history_years", 0) or 0) or float(keep) < 5):
        raise ConfigError("retention.price_history_years harus >= data.initial_history_years dan >= 5 "
                          "(walk-forward butuh ±4,7 tahun histori)")
    if get(cfg, "split.mode") not in ("rolling", "fixed"):
        raise ConfigError("split.mode harus rolling atau fixed")


def resolve_path(cfg: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg.get("_root", ROOT)) / p


def cache_dir(cfg: dict) -> Path:
    p = resolve_path(cfg, get(cfg, "project.cache_dir", "data/cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# kompatibilitas modul lama
artifacts_dir = cache_dir
data_dir = cache_dir


def embargo_days(cfg: dict) -> int:
    e = get(cfg, "split.embargo_days")
    return int(e) if e is not None else int(max(get(cfg, "labels.HORIZONS")))


def snapshot(cfg: dict) -> dict:
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_") and k not in ("database",)}


def config_hash(cfg: dict) -> str:
    """Hash parameter yang memengaruhi model/strategi (tanpa secrets/URL) — disimpan di setiap model version."""
    keys = ["labels", "split", "model", "setups", "strategy", "scoring", "regime", "liquidity", "quality"]
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def is_production(cfg: dict) -> bool:
    return str(get(cfg, "environment", "development")).lower() == "production"
