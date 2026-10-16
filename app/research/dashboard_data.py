"""Logika data untuk 4 halaman dashboard riset (dapat dites tanpa Streamlit).

Semua fungsi hanya MEMBACA database. Data yang tidak tersedia ditampilkan dengan labelnya, bukan nol.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.database.repository import Repository
from app.research.integrated_scoring import RANKINGS, assemble, rank_all

NUM = ["price", "fair_value_low", "fair_value_base", "fair_value_high", "margin_of_safety", "margin_of_safety_conservative",
       "value_score", "quality_score"]


def latest_prediction_date(repo: Repository):
    d = repo.db.scalar("SELECT MAX(prediction_date) FROM predictions")
    return None if d is None else str(d)[:10]


def undervalued_screener(repo: Repository, min_mos: float = 0.0, statuses=("DEEP_VALUE", "UNDERVALUED"),
                         exclude_trap_high: bool = True, sector_types=None, min_quality: float | None = None) -> pd.DataFrame:
    d = repo.db.scalar("SELECT MAX(as_of_date) FROM valuation_results")
    if d is None:
        return pd.DataFrame()
    v = repo.db.query_df("SELECT s.ticker, s.name, s.sector, v.* FROM valuation_results v JOIN stocks s ON s.id = v.stock_id "
                         "WHERE v.as_of_date = ?", (str(d)[:10],))
    for c in NUM:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v = v[v["valuation_status"].isin(list(statuses))]
    v = v[v["margin_of_safety"] >= min_mos]
    if exclude_trap_high:
        v = v[v["value_trap_risk"] != "HIGH"]
    if sector_types:
        v = v[v["sector_type"].isin(list(sector_types))]
    if min_quality is not None:
        v = v[v["quality_score"] >= min_quality]
    v["trap_reasons"] = v["value_trap_reasons"].map(lambda s: "; ".join((json.loads(s) or {}).get("reasons", [])[:3])
                                                    if isinstance(s, str) else "")
    cols = ["ticker", "name", "sector", "sector_type", "price", "fair_value_low", "fair_value_base", "fair_value_high",
            "margin_of_safety", "margin_of_safety_conservative", "valuation_status", "valuation_confidence", "value_score",
            "quality_score", "value_trap_risk", "trap_reasons", "fundamentals_period_end", "fundamentals_known_date"]
    return v.reindex(columns=cols).sort_values("value_score", ascending=False).reset_index(drop=True)


def valuation_coverage(repo: Repository) -> dict:
    d = repo.db.scalar("SELECT MAX(as_of_date) FROM valuation_results")
    if d is None:
        return {"date": None, "total": 0}
    r = repo.db.query_df("SELECT valuation_status, data_status, COUNT(*) AS n FROM valuation_results WHERE as_of_date = ? "
                         "GROUP BY valuation_status, data_status", (str(d)[:10],))
    return {"date": str(d)[:10], "total": int(r["n"].sum()),
            "by_status": r.groupby("valuation_status")["n"].sum().to_dict(),
            "by_data_status": r.groupby("data_status")["n"].sum().to_dict()}


def source_status(repo: Repository) -> pd.DataFrame:
    return repo.db.query_df("SELECT name, category, status, detail, checked_at FROM data_sources "
                            "WHERE category IN ('fundamental', 'foreign_flow') ORDER BY category, priority")


def foreign_flow_overview(repo: Repository, days: int = 60) -> dict:
    last = repo.db.scalar("SELECT MAX(date) FROM foreign_flow_history")
    if last is None:
        return {"status": "FOREIGN_FLOW_UNAVAILABLE"}
    start = (pd.Timestamp(last) - pd.Timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
    f = repo.db.query_df("SELECT s.ticker, f.date, f.net_foreign_shares, f.net_foreign_value, f.value_type, "
                         "f.foreign_sell_shares, f.total_volume_shares "
                         "FROM foreign_flow_history f JOIN stocks s ON s.id = f.stock_id WHERE f.date >= ?", (start,),
                         parse_dates=["date"])
    px = repo.db.query_df("SELECT s.ticker, p.date, p.close, p.value, p.volume FROM price_history p JOIN stocks s "
                          "ON s.id = p.stock_id WHERE p.date >= ?", (start,), parse_dates=["date"])
    for c in ("net_foreign_shares", "net_foreign_value", "foreign_sell_shares", "total_volume_shares"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    for c in ("close", "value", "volume"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    m = f.merge(px, on=["ticker", "date"], how="left")
    vwap = (m["value"] / m["volume"]).where(m["volume"] > 0).fillna(m["close"])
    m["net_value"] = m["net_foreign_value"].where(m["net_foreign_value"].notna(), m["net_foreign_shares"] * vwap)
    m["is_estimated"] = m["net_foreign_value"].isna()
    daily = m.groupby("date").agg(net_value=("net_value", "sum"), estimated_share=("is_estimated", "mean"),
                                  short_shares=("foreign_sell_shares", "sum"),
                                  reported_volume=("total_volume_shares", "sum")).reset_index()
    # rasio short volume FINRA seluruh universe (Σ short / Σ volume yang dilaporkan ke FINRA)
    daily["short_ratio"] = (daily["short_shares"] / daily["reported_volume"].where(daily["reported_volume"] > 0))
    return {"status": "AVAILABLE", "last_date": str(last)[:10], "daily": daily,
            "value_label": "ESTIMATED_VALUE" if m["is_estimated"].any() else "ACTUAL_VALUE"}


def ticker_flows(repo: Repository, ticker: str, days: int = 120) -> pd.DataFrame:
    df = repo.db.query_df("SELECT f.date, f.foreign_sell_shares AS short_volume, f.total_volume_shares AS finra_volume, "
                          "f.foreign_buy_shares AS non_short_volume, f.net_foreign_shares AS net_shares, f.source, "
                          "f.quality_status FROM foreign_flow_history f "
                          "JOIN stocks s ON s.id = f.stock_id WHERE s.ticker = ? ORDER BY f.date DESC LIMIT ?",
                          (ticker, days), parse_dates=["date"]).sort_values("date")
    for c in ("short_volume", "finra_volume", "non_short_volume", "net_shares"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["short_ratio"] = df["short_volume"] / df["finra_volume"].where(df["finra_volume"] > 0)
    return df


def prediction_research(repo: Repository, date: str | None = None) -> pd.DataFrame:
    d = date or latest_prediction_date(repo)
    if d is None:
        return pd.DataFrame()
    df = repo.db.query_df("SELECT p.ticker, s.name, s.sector, p.decision, p.setup, p.score, p.close_price, p.value_score, "
                          "p.quality_score, p.value_trap_risk, p.margin_of_safety, p.valuation_status, p.foreign_flow_score, "
                          "p.foreign_flow_status, p.accumulation_score, p.distribution_risk, p.accumulation_status, "
                          "p.accumulation_stage FROM predictions p JOIN stocks s ON s.id = p.stock_id "
                          "WHERE p.prediction_date = ?", (d,))
    for c in ("score", "close_price", "value_score", "quality_score", "margin_of_safety", "foreign_flow_score",
              "accumulation_score", "distribution_risk"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.attrs["date"] = d
    return df


def foreign_flow_table(repo: Repository) -> pd.DataFrame:
    p = prediction_research(repo)
    if p.empty:
        return p
    return p[["ticker", "name", "sector", "foreign_flow_status", "foreign_flow_score", "accumulation_status", "decision"]] \
        .sort_values("foreign_flow_score", ascending=False, na_position="last").reset_index(drop=True)


def accumulation_table(repo: Repository, stages=None) -> pd.DataFrame:
    p = prediction_research(repo)
    if p.empty:
        return p
    if stages:
        p = p[p["accumulation_stage"].isin(list(stages))]
    return p[["ticker", "name", "sector", "accumulation_stage", "accumulation_status", "accumulation_score",
              "distribution_risk", "foreign_flow_status", "decision", "setup"]] \
        .sort_values("accumulation_score", ascending=False, na_position="last").reset_index(drop=True)


def accumulation_evidence(repo: Repository, ticker: str, limit: int = 30) -> pd.DataFrame:
    df = repo.db.query_df("SELECT a.date, a.accumulation_score, a.distribution_risk, a.status, a.stage, a.confidence, "
                          "a.evidence FROM accumulation_signals a JOIN stocks s ON s.id = a.stock_id WHERE s.ticker = ? "
                          "ORDER BY a.date DESC LIMIT ?", (ticker, limit))
    df["evidence"] = df["evidence"].map(lambda s: "; ".join((json.loads(s) or {}).get("evidence", []))
                                        if isinstance(s, str) else "")
    return df


def integrated_table(repo: Repository, cfg: dict) -> pd.DataFrame:
    """Ranking terintegrasi dari hasil daily terakhir (tanpa menghitung ulang modul berat)."""
    from app.research.report import _signals_from_db
    from app.valuation.valuation_report import latest_valuations
    d = latest_prediction_date(repo)
    if d is None:
        return pd.DataFrame()
    sig = _signals_from_db(repo, d)
    val = latest_valuations(repo, d)
    for c in NUM:
        if c in val:
            val[c] = pd.to_numeric(val[c], errors="coerce")
    p = prediction_research(repo, d)
    flow = p[["ticker", "foreign_flow_score", "foreign_flow_status"]].assign(foreign_flow_confidence=None)
    acc = p[["ticker", "accumulation_score", "distribution_risk", "accumulation_status", "accumulation_stage"]] \
        .assign(accumulation_confidence=None)
    t = rank_all(assemble(sig, val if len(val) else None, flow, acc), cfg)
    t.attrs["date"] = d
    return t


def ranking_view(t: pd.DataFrame, name: str, n: int = 20) -> pd.DataFrame:
    if name not in RANKINGS or f"rank_{name}" not in t:
        return pd.DataFrame()
    cols = ["ticker", "name", "sector", f"score_{name}", "value_decision", "swing_decision", "valuation_status",
            "value_trap_risk", "foreign_flow_status", "accumulation_status", "value_score", "business_quality_score",
            "accumulation_score", "foreign_flow_score", "technical_setup_score", "ml_probability_score"]
    out = t[t[f"rank_{name}"].notna()].sort_values(f"rank_{name}").head(n)
    return out.reindex(columns=cols).replace({np.nan: None}).reset_index(drop=True)
