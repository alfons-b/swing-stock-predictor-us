"""Evaluasi historis modul riset: `python main.py evaluate-modules`.

Kejujuran metodologi:
- Sinyal swing dasar = pipeline sinyal IDENTIK dengan scan, tetapi dengan prediksi NETRAL (teknikal-saja).
  Model ML aktif dilatih pada data hingga baru-baru ini → memakainya pada periode lampau = kebocoran.
  (Kontribusi ML dievaluasi terpisah oleh walk-forward backtest yang melatih ulang per fold.)
- Semua ambang varian berasal dari config (ditetapkan sebelum evaluasi). Periode VALIDATION dan TEST dilaporkan
  terpisah; test tidak dipakai untuk memilih varian/ambang.
- Valuasi historis point-in-time: dihitung ulang tiap akhir bulan memakai laporan yang first_known_date ≤ tanggal itu.
  Tanpa histori fundamental PIT (mis. hanya snapshot Yahoo yang baru diambil) → INSUFFICIENT_HISTORY. Mode
  `--estimated-pit` memakai estimasi tanggal publikasi dan SELURUH hasilnya berlabel ESTIMATED_PIT (riset saja).
- Varian yang data pendukungnya < 50% sinyal → INSUFFICIENT_DATA, bukan angka.
"""
from __future__ import annotations

import copy
import uuid

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestEngine
from app.backtest.runner import neutral_predictions, technical_only_cfg
from app.backtest.walk_forward import make_folds
from app.config import config_hash, get
from app.database.repository import Repository
from app.risk.concentration import apply_concentration_limits
from app.scanner.ranking import generate_signals, rank_signals
from app.utils.logging_utils import get_logger

log = get_logger(__name__)
MIN_COVERAGE = 0.5
MIN_IC_DATES = 12

VARIANTS = {
    "V1_swing_baseline": "sinyal swing teknikal (tanpa modul riset)",
    "V2_excl_distribution": "V1 tanpa BUY ber-distribution risk ≥ 60",
    "V3_with_accumulation": "V1 hanya bila accumulation_score ≥ 50",
    "V4_with_foreign_flow": "V1 tanpa BUY saat short flow FINRA NET_SELLING/STRONG_NET_SELLING (butuh histori flow)",
    "V5_excl_value_trap": "V1 tanpa BUY ber-value_trap_risk HIGH",
    "V6_with_value": "V1 hanya bila value_score ≥ 50",
    "V7_with_quality": "V1 hanya bila business_quality_score ≥ 50",
    "V8_integrated_rank": "V1 diurutkan ulang dengan composite integrated (bukan rank_score)",
    "V9_value_accumulation": "V1 hanya bila valuasi UNDERVALUED/DEEP_VALUE dan accumulation_score ≥ 50",
}
REQUIRES = {"V2_excl_distribution": "distribution_risk_score", "V3_with_accumulation": "accumulation_score",
            "V4_with_foreign_flow": "foreign_flow_score", "V5_excl_value_trap": "value_trap_known",
            "V6_with_value": "value_score", "V7_with_quality": "business_quality_score",
            "V8_integrated_rank": "score_integrated", "V9_value_accumulation": "value_score"}


# --------------------------------------------------------------------------------------- overlay historis
def accumulation_history(cfg: dict, repo: Repository, df: pd.DataFrame) -> pd.DataFrame:
    from app.accumulation.accumulation_indicators import compute_indicators
    from app.accumulation.accumulation_scorer import score
    from app.flows.flow_features import build_flow_features
    from app.flows.flow_scorer import score_flows
    from app.flows.foreign_flow_provider import load_flows
    px = df[["ticker", "date", "open", "high", "low", "close", "volume", "value"]]
    flows = load_flows(repo, start=px["date"].min(), end=px["date"].max())
    fl = None
    if len(flows):
        feat = build_flow_features(px, flows, baseline_days=get(cfg, "foreign_flow.baseline_days"))
        fl = score_flows(feat, float(get(cfg, "foreign_flow.min_coverage", 0.8)))
    ind = compute_indicators(px)
    sc = score(ind, fl, (cfg.get("accumulation") or {}).get("thresholds"))
    out = sc[["ticker", "date", "accumulation_score", "distribution_risk", "accumulation_status", "accumulation_stage"]]
    if fl is not None:
        out = out.merge(fl[["ticker", "date", "foreign_flow_score", "foreign_flow_status"]], on=["ticker", "date"], how="left")
    else:
        out = out.assign(foreign_flow_score=np.nan, foreign_flow_status="FOREIGN_FLOW_UNAVAILABLE")
    return out.rename(columns={"distribution_risk": "distribution_risk_score"})


def valuation_history(cfg: dict, repo: Repository, df: pd.DataFrame, start, end, use_estimated: bool) -> pd.DataFrame:
    from app.valuation.fundamental_provider import load_statements
    from app.valuation.valuation_report import run_valuation
    st = load_statements(repo, None, use_estimated=use_estimated)
    if st.empty:
        return pd.DataFrame()
    first_usable = st["usable_from"].min()
    dates = pd.Series(sorted(df["date"].unique()))
    dates = dates[(dates >= max(pd.Timestamp(start), first_usable)) & (dates <= pd.Timestamp(end))]
    month_ends = dates.groupby(dates.dt.to_period("M")).max().tolist()
    prices = df[["ticker", "date", "close", "open", "high", "low", "volume", "value"]]
    frames = []
    for d in month_ends:
        v = run_valuation(cfg, repo, d, prices[prices["date"] <= d], use_estimated=use_estimated)
        if len(v):
            frames.append(v.assign(as_of=pd.Timestamp(d)))
    if not frames:
        return pd.DataFrame()
    v = pd.concat(frames, ignore_index=True)
    return v[["ticker", "as_of", "value_score", "quality_score", "value_trap_risk", "valuation_status",
              "margin_of_safety"]].rename(columns={"quality_score": "business_quality_score"})


def attach_overlays(sig: pd.DataFrame, acc: pd.DataFrame | None, val: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    from app.research.integrated_scoring import TRAP_INV, composite
    s = sig.copy()
    if acc is not None and len(acc):
        s = s.merge(acc, on=["ticker", "date"], how="left")
    for c in ("accumulation_score", "distribution_risk_score", "foreign_flow_score"):
        if c not in s:
            s[c] = np.nan
    if val is not None and len(val):
        s = s.assign(ticker=s["ticker"].astype(object)).sort_values("date")
        v = val.assign(ticker=val["ticker"].astype(object)).sort_values("as_of")
        s = pd.merge_asof(s, v, left_on="date", right_on="as_of", by="ticker", direction="backward",
                          tolerance=pd.Timedelta(days=40))
    for c in ("value_score", "business_quality_score", "margin_of_safety"):
        if c not in s:
            s[c] = np.nan
    if "value_trap_risk" not in s:
        s["value_trap_risk"] = None
    if "valuation_status" not in s:
        s["valuation_status"] = None
    s["value_trap_known"] = s["value_trap_risk"].map(lambda r: 1.0 if r in ("LOW", "MEDIUM", "HIGH") else np.nan)
    s["value_risk_inverse"] = s["value_trap_risk"].map(TRAP_INV)
    s["distribution_risk_inverse"] = 100 - s["distribution_risk_score"]
    s["technical_setup_score"] = s.get("technical_score")
    s["ml_probability_score"] = np.nan                     # teknikal-saja: tanpa ML (lihat docstring)
    s["market_regime_score"] = s["market_regime"].astype(object).map(
        __import__("app.strategy.scoring", fromlist=["REGIME_SCORE"]).REGIME_SCORE)
    w = get(cfg, "research_scoring.rankings.integrated", {}) or {}
    s["score_integrated"] = composite(s, w, float(get(cfg, "research_scoring.min_weight_coverage", 0.6)))
    return s.sort_values(["date", "ticker"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ varian
def apply_variant(name: str, s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    buy = s["decision"] == "BUY"
    drop = pd.Series(False, index=s.index)
    if name == "V2_excl_distribution":
        drop = s["distribution_risk_score"] >= 60
    elif name == "V3_with_accumulation":
        drop = ~(s["accumulation_score"] >= 50)
    elif name == "V4_with_foreign_flow":
        drop = s["foreign_flow_status"].isin(["NET_SELLING", "STRONG_NET_SELLING"]) if "foreign_flow_status" in s else drop
    elif name == "V5_excl_value_trap":
        drop = s["value_trap_risk"] == "HIGH"
    elif name == "V6_with_value":
        drop = ~(s["value_score"] >= 50)
    elif name == "V7_with_quality":
        drop = ~(s["business_quality_score"] >= 50)
    elif name == "V8_integrated_rank":
        s["rank_score"] = s["score_integrated"].where(s["score_integrated"].notna(), s["rank_score"] * 0.5)
    elif name == "V9_value_accumulation":
        drop = ~(s["valuation_status"].isin(["UNDERVALUED", "DEEP_VALUE"]) & (s["accumulation_score"] >= 50))
    s.loc[buy & drop.fillna(False), "decision"] = "WAIT"
    return s


def _coverage(name: str, s: pd.DataFrame) -> float:
    col = REQUIRES.get(name)
    if col is None:
        return 1.0
    b = s[s["decision"] == "BUY"]
    return float(b[col].notna().mean()) if len(b) else 0.0


def breakdown(trades: pd.DataFrame, s: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {}
    t = trades.merge(s[["ticker", "date", "market_regime", "sector", "setup_type", "valuation_status",
                        "accumulation_status"]].rename(columns={"date": "signal_date"}), on=["ticker", "signal_date"],
                     how="left")
    out = {}
    for col in ("market_regime", "sector", "setup_type", "valuation_status", "accumulation_status"):
        g = t.groupby(t[col].fillna("UNAVAILABLE"))["net_return"]
        out[col] = {str(k): {"trades": int(v.count()), "win_rate": round(float((v > 0).mean()), 3),
                             "avg_net_return": round(float(v.mean()), 4)} for k, v in g}
    return out


def _metrics(m: dict) -> dict:
    keys = ("total_return", "cagr", "max_drawdown", "sharpe", "number_of_trades", "win_rate", "expectancy_pct",
            "profit_factor", "average_holding_days")
    return {k: m.get(k) for k in keys if k in m}


# --------------------------------------------------------------------------------------------------- IC
def factor_ic(s: pd.DataFrame, df: pd.DataFrame, factor: str, horizons, sample_every: int = 1) -> dict:
    """Rank IC (Spearman) faktor vs return ke depan per tanggal: rata-rata, t-stat, jumlah tanggal."""
    px = df[["ticker", "date", "close"]].sort_values(["ticker", "date"])
    g = px.groupby("ticker")["close"]
    for h in horizons:
        px[f"fwd_{h}"] = g.shift(-h) / px["close"] - 1
    x = s[["ticker", "date", factor]].dropna().merge(px, on=["ticker", "date"], how="inner")
    dates = sorted(x["date"].unique())[::max(1, sample_every)]
    x = x[x["date"].isin(dates)]
    out = {}
    for h in horizons:
        ics = []
        for _, grp in x.dropna(subset=[f"fwd_{h}"]).groupby("date"):
            if len(grp) >= 10 and grp[factor].nunique() > 2:
                ics.append(grp[factor].rank().corr(grp[f"fwd_{h}"].rank()))
        ics = pd.Series(ics).dropna()
        if len(ics) < MIN_IC_DATES:
            out[f"{h}d"] = {"status": "INSUFFICIENT_HISTORY", "n_dates": int(len(ics))}
        else:
            out[f"{h}d"] = {"status": "OK", "mean_ic": round(float(ics.mean()), 4),
                            "t_stat": round(float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), 2)
                            if ics.std(ddof=1) > 0 else None,
                            "pct_positive": round(float((ics > 0).mean()), 3), "n_dates": int(len(ics)),
                            "note": "tanggal bertumpuk (overlapping) → t-stat optimistis" if sample_every < h else ""}
    return out


def redundancy_vs_momentum(s: pd.DataFrame, df: pd.DataFrame, factor: str, n: int = 20, sample_every: int = 5) -> dict:
    """Rata-rata korelasi rank lintas-emiten antara faktor dan momentum n hari. Tinggi (> 0,7) = faktor sebagian besar
    hanya mengulang momentum — IC-nya belum tentu informasi baru."""
    px = df[["ticker", "date", "close"]].sort_values(["ticker", "date"]).copy()
    px["mom"] = px.groupby("ticker")["close"].pct_change(n)
    x = s[["ticker", "date", factor]].dropna().merge(px[["ticker", "date", "mom"]].dropna(), on=["ticker", "date"])
    dates = sorted(x["date"].unique())[::max(1, sample_every)]
    cs = [g[factor].rank().corr(g["mom"].rank()) for _, g in x[x["date"].isin(dates)].groupby("date") if len(g) >= 10]
    cs = pd.Series(cs).dropna()
    return {"mean_rank_corr_with_momentum": round(float(cs.mean()), 3) if len(cs) else None, "n_dates": int(len(cs))}


# -------------------------------------------------------------------------------------------------- utama
def evaluate_modules(cfg: dict, repo: Repository, df: pd.DataFrame | None = None, use_estimated: bool = False,
                     save: bool = True) -> dict:
    from app.pipeline.market_data import build_dataset
    if df is None:
        df = build_dataset(cfg, repo, None, labels=False)
    folds = make_folds(df["date"], cfg)
    periods = {}
    val = [f for f in folds if f.kind == "validation"]
    if val:
        periods["validation"] = (min(f.eval_start for f in val), max(f.eval_end for f in val))
    test = [f for f in folds if f.kind == "test"]
    if test:
        periods["test"] = (test[0].eval_start, test[0].eval_end)
    if not periods:
        return {"status": "INSUFFICIENT_HISTORY", "reason": "tidak ada fold validasi/test (histori terlalu pendek)"}
    tcfg = technical_only_cfg(cfg)
    start_all = min(p[0] for p in periods.values())
    feat = df[df["date"] >= start_all - pd.Timedelta(days=10)].reset_index(drop=True)
    sig = generate_signals(feat, neutral_predictions(feat.index, cfg["labels"]["SWING_HORIZON"]), tcfg)
    log.info("Evaluasi modul: overlay akumulasi/flow ...")
    acc = accumulation_history(cfg, repo, df)
    log.info("Evaluasi modul: valuasi point-in-time bulanan ...")
    vh = valuation_history(cfg, repo, df, start_all - pd.DateOffset(months=13), df["date"].max(), use_estimated)
    s = attach_overlays(sig, acc, vh, cfg)
    label = "ESTIMATED_PIT" if use_estimated else "PIT"
    result = {"status": "OK", "pit_mode": label, "baseline": "technical-only (prediksi netral)", "periods": {},
              "variants": VARIANTS, "selection_rule": "ambang dari config, ditetapkan sebelum evaluasi; test tidak dipakai memilih",
              "factor_ic": {}}
    # IC faktor (horizon valuasi panjang; akumulasi/flow pendek)
    if len(vh):
        vv = vh.rename(columns={"as_of": "date"})
        result["factor_ic"]["value_score"] = factor_ic(vv, df, "value_score", (60, 120, 250))
        result["factor_ic"]["business_quality_score"] = factor_ic(vv, df, "business_quality_score", (60, 120, 250))
    else:
        result["factor_ic"]["value_score"] = {"status": "INSUFFICIENT_HISTORY",
                                              "reason": "tidak ada laporan keuangan point-in-time di database"}
    result["factor_ic"]["accumulation_score"] = factor_ic(acc, df, "accumulation_score", (5, 10, 20), sample_every=5)
    result["factor_ic"]["momentum_20d (pembanding)"] = factor_ic(
        df[["ticker", "date", "close"]].sort_values(["ticker", "date"]).assign(
            momentum_20d=lambda x: x.groupby("ticker")["close"].pct_change(20)), df, "momentum_20d", (5, 10, 20),
        sample_every=5)
    result["redundancy"] = {"accumulation_score": redundancy_vs_momentum(acc, df, "accumulation_score")}
    if acc["foreign_flow_score"].notna().any():
        result["factor_ic"]["foreign_flow_score"] = factor_ic(acc, df, "foreign_flow_score", (5, 10, 20), sample_every=5)
    else:
        result["factor_ic"]["foreign_flow_score"] = {"status": "FOREIGN_FLOW_UNAVAILABLE"}
    for pname, (ps, pe) in periods.items():
        rows = {}
        for v in VARIANTS:
            sv = apply_variant(v, s[(s["date"] >= ps) & (s["date"] <= pe)])
            cov = _coverage(v, s[(s["date"] >= ps) & (s["date"] <= pe)])
            if cov < MIN_COVERAGE:
                rows[v] = {"status": "INSUFFICIENT_DATA", "data_coverage": round(cov, 3)}
                continue
            ranked = rank_signals(apply_concentration_limits(sv, tcfg, df), tcfg)
            res = BacktestEngine(copy.deepcopy(tcfg)).run(ranked, df, ps, pe)
            rows[v] = {"status": "OK", "data_coverage": round(cov, 3), "metrics": _metrics(res["metrics"]),
                       "breakdown": breakdown(res["trades"], sv)}
        base = rows.get("V1_swing_baseline", {}).get("metrics", {})
        for v, r in rows.items():
            if r.get("status") == "OK" and base:
                r["vs_baseline"] = {k: (None if r["metrics"].get(k) is None or base.get(k) is None
                                        else round(r["metrics"][k] - base[k], 4))
                                    for k in ("total_return", "sharpe", "max_drawdown", "win_rate")}
        result["periods"][pname] = {"start": str(pd.Timestamp(ps).date()), "end": str(pd.Timestamp(pe).date()), "variants": rows}
    if save:
        run_id = f"modules-{pd.Timestamp.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
        first = next(iter(result["periods"].values()))
        repo.save_backtest(run_id, "module_evaluation", {"start": first["start"], "end": result["periods"][list(result["periods"])[-1]]["end"],
                                                         **result}, {}, None, "technical_only", config_hash(cfg))
        result["run_id"] = run_id
    return result


def _ic_text(v: dict) -> str:
    if v.get("status") != "OK":
        return v.get("status", "-")
    return f"IC {v['mean_ic']:+.3f} (t {v.get('t_stat')}, n {v['n_dates']})"


def format_evaluation(res: dict) -> str:
    if res.get("status") != "OK":
        return f"Evaluasi modul: {res.get('status')} — {res.get('reason', '')}"
    L = [f"EVALUASI MODUL RISET ({res['pit_mode']}; baseline {res['baseline']})", res["selection_rule"], ""]
    for f, r in res["factor_ic"].items():
        if "status" in r and len(r) <= 2:
            L.append(f"IC {f:<24}: {r['status']} {r.get('reason', '')}")
            continue
        parts = [f"{h}: {_ic_text(v)}" for h, v in r.items()]
        L.append(f"IC {f:<24}: " + " | ".join(parts))
    for f, r in (res.get("redundancy") or {}).items():
        L.append(f"Korelasi {f} vs momentum 20H: {r.get('mean_rank_corr_with_momentum')} "
                 "(> 0,7 = sebagian besar mengulang momentum)")
    for pname, p in res["periods"].items():
        L += ["", f"[{pname.upper()}] {p['start']} .. {p['end']}",
              f"{'varian':<24}{'status':<18}{'return':>9}{'sharpe':>8}{'maxDD':>8}{'trades':>8}{'win%':>7}"]
        for v, r in p["variants"].items():
            if r["status"] != "OK":
                L.append(f"{v:<24}{r['status']:<18} (cakupan data {r['data_coverage']:.0%})")
                continue
            m = r["metrics"]
            L.append(f"{v:<24}{'OK':<18}{m.get('total_return', 0):>+9.1%}{m.get('sharpe', 0):>8.2f}"
                     f"{m.get('max_drawdown', 0):>8.1%}{m.get('number_of_trades', 0):>8d}{(m.get('win_rate') or 0):>7.0%}")
    L += ["", "Catatan: hasil historis bukan jaminan; selisih kecil antar varian umumnya tidak signifikan secara statistik."]
    return "\n".join(L)
