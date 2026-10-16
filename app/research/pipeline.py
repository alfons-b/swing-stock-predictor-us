"""Orkestrasi modul riset dalam daily pipeline.

Urutan: fundamental (incremental) → foreign flow (incremental) → valuasi → fitur & skor flow → akumulasi →
skor terintegrasi & ranking. Setiap tahap terisolasi: kegagalan modul riset dicatat (status) dan TIDAK
menggagalkan pipeline swing. Data yang tidak tersedia diberi label (FOREIGN_FLOW_UNAVAILABLE, INSUFFICIENT_DATA).
"""
from __future__ import annotations

import pandas as pd

from app.config import get
from app.database.repository import Repository
from app.research.integrated_scoring import assemble, rank_all
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

PRED_RESEARCH_COLS = {"value_score": "value_score", "quality_score": "business_quality_score",
                      "value_trap_risk": "value_trap_risk", "margin_of_safety": "margin_of_safety",
                      "valuation_status": "valuation_status", "foreign_flow_score": "foreign_flow_score",
                      "foreign_flow_status": "foreign_flow_status", "accumulation_score": "accumulation_score",
                      "distribution_risk": "distribution_risk_score", "accumulation_status": "accumulation_status",
                      "accumulation_stage": "accumulation_stage"}


def _stage(name, fn, status: dict):
    try:
        res = fn()
        status[name] = "OK"
        return res
    except Exception as e:  # modul riset tidak boleh menjatuhkan pipeline swing
        log.warning("Modul riset %s gagal: %s", name, e, exc_info=True, extra={"persist": True})
        status[name] = f"FAILED: {str(e)[:200]}"
        return None


def run_research(cfg: dict, repo: Repository, prices: pd.DataFrame, sig: pd.DataFrame, as_of, run_id: str | None = None,
                 update_sources: bool = True) -> dict:
    from app.accumulation.accumulation_report import run_accumulation, run_flow_analysis, save_accumulation
    from app.flows.foreign_flow_provider import update_foreign_flow
    from app.valuation.fundamental_provider import update_fundamentals
    from app.valuation.valuation_report import run_valuation, save_valuations

    as_of = pd.Timestamp(as_of).normalize()
    status, stats = {}, {}
    if update_sources:
        stats["fundamentals"] = _stage("fundamentals", lambda: update_fundamentals(cfg, repo, as_of), status)
        stats["foreign_flow"] = _stage("foreign_flow_update", lambda: update_foreign_flow(cfg, repo, as_of), status)
    flow_state = (stats.get("foreign_flow") or {}).get("status")
    if flow_state is None:
        flow_state = "AVAILABLE" if repo.db.scalar("SELECT COUNT(*) FROM foreign_flow_history") else "FOREIGN_FLOW_UNAVAILABLE"
    val = None
    if get(cfg, "valuation.enabled", True):
        val = _stage("valuation", lambda: run_valuation(cfg, repo, as_of, prices, run_id=run_id), status)
        if val is not None:
            _stage("valuation_save", lambda: save_valuations(repo, val), status)
    flow = _stage("flow_analysis", lambda: run_flow_analysis(cfg, repo, prices, as_of), status) \
        if get(cfg, "foreign_flow.enabled", True) else None
    flow_last = None
    if flow is not None and len(flow):
        flow_last = flow.sort_values("date").groupby("ticker").tail(1)
        flow_last = flow_last[flow_last["date"] >= as_of - pd.Timedelta(days=7)]   # flow basi tidak dipakai
    acc = None
    if get(cfg, "accumulation.enabled", True):
        acc = _stage("accumulation", lambda: run_accumulation(cfg, repo, prices, as_of, flow), status)
        if acc is not None:
            stats["accumulation_saved"] = _stage("accumulation_save", lambda: save_accumulation(cfg, repo, acc, run_id), status)
    table = _stage("integrated_scoring", lambda: rank_all(assemble(sig, val, flow_last, acc), cfg), status)
    summary = {"status": status, "foreign_flow_status": flow_state,
               "valuation_ok": int((val["valuation_status"] != "INSUFFICIENT_DATA").sum()) if val is not None and len(val) else 0,
               "valuation_total": 0 if val is None else len(val),
               "flow_scored": 0 if flow_last is None else int(flow_last["foreign_flow_score"].notna().sum()),
               "accumulation_signals": int((acc["accumulation_status"].str.contains("ACCUMULATION")).sum())
               if acc is not None and len(acc) else 0}
    if table is not None:
        summary["value_candidates"] = table.loc[table["value_decision"] == "VALUE_CANDIDATE", "ticker"].tolist()[:20]
    log.info("Riset: valuasi %d/%d, foreign flow %s (%d emiten dinilai), sinyal akumulasi %d", summary["valuation_ok"],
             summary["valuation_total"], flow_state, summary["flow_scored"], summary["accumulation_signals"],
             extra={"persist": True})
    return {"table": table, "valuation": val, "flow": flow_last, "accumulation": acc, "summary": summary, "stats": stats}


def attach_to_signals(sig: pd.DataFrame, table: pd.DataFrame | None) -> pd.DataFrame:
    """Tambahkan kolom riset ke sinyal (disimpan di tabel predictions). Tidak mengubah keputusan swing."""
    if table is None or table.empty:
        return sig
    m = table.drop_duplicates("ticker").set_index("ticker")
    sig = sig.copy()                                            # index dipertahankan (dipakai summarize_scan)
    for k, v in PRED_RESEARCH_COLS.items():
        if v in m:
            sig[f"research_{k}"] = sig["ticker"].map(m[v])
    return sig
