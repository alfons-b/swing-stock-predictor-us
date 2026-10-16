"""Orkestrasi foreign flow + akumulasi/distribusi untuk satu tanggal, penyimpanan, dan teks laporan."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from app.accumulation.accumulation_indicators import compute_indicators
from app.accumulation.accumulation_scorer import evidence, score
from app.config import get
from app.database.repository import Repository
from app.flows.flow_features import build_flow_features
from app.flows.flow_scorer import score_flows
from app.flows.foreign_flow_provider import load_flows
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def run_flow_analysis(cfg: dict, repo: Repository, prices: pd.DataFrame, as_of) -> pd.DataFrame:
    """Fitur + skor foreign flow untuk semua emiten pada tanggal-tanggal di `prices` (≤ as_of)."""
    as_of = pd.Timestamp(as_of)
    p = prices[prices["date"] <= as_of]
    if p.empty:
        return pd.DataFrame()
    # cukup untuk jendela 60 hari bursa (+ baseline N hari bursa bila diaktifkan, ±1,5 hari kalender per hari bursa)
    base = int(get(cfg, "foreign_flow.baseline_days") or 0)
    start = p["date"].max() - pd.Timedelta(days=130 + int(base * 1.5))
    p = p[p["date"] >= start]
    flows = load_flows(repo, start=start, end=as_of)
    feat = build_flow_features(p, flows, tuple(get(cfg, "foreign_flow.windows", [5, 20, 60])),
                               get(cfg, "foreign_flow.baseline_days"))
    sc = score_flows(feat, float(get(cfg, "foreign_flow.min_coverage", 0.8)))
    return feat.merge(sc, on=["ticker", "date"], how="left")


def run_accumulation(cfg: dict, repo: Repository, prices: pd.DataFrame, as_of, flow: pd.DataFrame | None = None) -> pd.DataFrame:
    """Status akumulasi pada `as_of` (baris terakhir ≤ as_of tiap emiten)."""
    acfg = cfg.get("accumulation", {}) or {}
    as_of = pd.Timestamp(as_of)
    p = prices[prices["date"] <= as_of]
    p = p[p["date"] >= p["date"].max() - pd.Timedelta(days=200)]       # ±130 hari bursa: cukup untuk jendela 60
    ind = compute_indicators(p, int(acfg.get("window", 20)), int(acfg.get("long_window", 60)))
    if ind.empty:
        return pd.DataFrame()
    last = ind.sort_values("date").groupby("ticker").tail(1)
    fl = None
    if flow is not None and len(flow):
        fl = flow[["ticker", "date", "foreign_flow_score", "foreign_flow_status"]]
    res = score(last, fl, acfg.get("thresholds"), float(get(cfg, "liquidity.MIN_AVG_TRADING_VALUE", 1e9)) / 5)
    ind_rows = last.set_index("ticker").to_dict("index")
    res["evidence"] = [evidence(r, ind_rows.get(r["ticker"], {})) for r in res.to_dict("records")]
    res["components"] = res["_components"]
    res["indicators"] = [{k: _num(v) for k, v in ind_rows.get(t, {}).items() if k.startswith("acc_")} for t in res["ticker"]]
    return res.drop(columns=["_components"])


def _num(v):
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def save_accumulation(cfg: dict, repo: Repository, res: pd.DataFrame, run_id: str | None = None) -> int:
    if res is None or res.empty:
        return 0
    keep_neutral = bool(get(cfg, "accumulation.store_neutral", False))
    d = res if keep_neutral else res[~res["accumulation_status"].isin(["NEUTRAL", "INSUFFICIENT_DATA"])
                                     | (res["accumulation_stage"] != "NO_CLEAR_SIGNAL")]
    if d.empty:
        return 0
    ids = repo.stock_ids()
    rows = pd.DataFrame({
        "stock_id": d["ticker"].map(ids), "date": pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"),
        "accumulation_score": d["accumulation_score"], "distribution_risk": d["distribution_risk"],
        "status": d["accumulation_status"], "stage": d["accumulation_stage"], "confidence": d["accumulation_confidence"],
        "components": [json.dumps({k: _num(v) for k, v in (c or {}).items()}) for c in d["components"]],
        "evidence": [json.dumps({"evidence": e, "indicators": i}) for e, i in zip(d["evidence"], d["indicators"])],
        "foreign_flow_status": d["foreign_flow_status"], "run_id": run_id}).dropna(subset=["stock_id"])
    rows["stock_id"] = rows["stock_id"].astype(int)
    repo.db.upsert("accumulation_signals", rows, keys=["stock_id", "date"], count=False)
    return len(rows)


def format_accumulation(row: dict | None) -> list[str]:
    if not row or row.get("accumulation_status") in (None, "INSUFFICIENT_DATA"):
        return ["Status               : INSUFFICIENT_DATA (histori < 60 hari bursa)"]
    ind = row.get("indicators") or {}
    f = lambda k, fmt: ("UNAVAILABLE" if ind.get(k) is None else format(ind[k], fmt))  # noqa: E731
    lines = [f"Status               : {row['accumulation_status']}  (skor {row['accumulation_score']:.0f}, "
             f"confidence {row.get('accumulation_confidence')})",
             f"Tahap                : {row['accumulation_stage']}",
             f"Distribution risk    : {row['distribution_risk']:.0f}" if row.get("distribution_risk") is not None
             and not pd.isna(row.get("distribution_risk")) else "Distribution risk    : UNAVAILABLE",
             f"CMF20 / MFI14        : {f('acc_cmf_20', '+.2f')} / {f('acc_mfi_14', '.0f')}",
             f"OBV / ADL trend 20H  : {f('acc_obv_trend_20', '+.2f')} / {f('acc_adl_trend_20', '+.2f')}",
             f"RVOL / harga vs VWAP : {f('acc_rvol_20', '.2f')} / {f('acc_price_vs_vwap', '+.1%')}"]
    for e in row.get("evidence") or []:
        lines.append(f"  - {e}")
    lines.append("  Catatan: bukti perilaku harga-volume, bukan identitas pembeli.")
    return lines
