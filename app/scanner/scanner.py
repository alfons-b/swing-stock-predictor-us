"""Daily scanner (§39) & analisis per saham (§59, §60).

Semua angka berasal dari database / provider / model (§61). Tidak ada yang dikarang: field yang
datanya tidak tersedia ditampilkan sebagai 'n/a'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import get
from app.data.tickers import normalize_ticker
from app.risk.concentration import apply_concentration_limits
from app.scanner.coverage import apply_coverage_gate
from app.scanner.ranking import generate_signals, rank_signals
from app.strategy.explain import build_reasoning, model_drivers

DISCLAIMER = ("ANALYTICAL DECISION SUPPORT SYSTEM — bukan sistem profit terjamin. Probabilitas berasal dari model "
              "historis dan bisa salah. Bukan nasihat investasi.")
SYNTHETIC_WARNING = "DATA CONTOH SINTETIS — hanya untuk menguji pipeline, BUKAN rekomendasi saham."


def apply_freshness_gate(sig: pd.DataFrame, freshness: dict) -> pd.DataFrame:
    if freshness.get("trading_allowed", True):
        return sig
    sig = sig.copy()
    buy = sig["decision"] == "BUY"
    sig.loc[buy, "decision"] = "WAIT"
    sig["reject_reasons"] = [list(r) + ["data_stale"] if b else r for r, b in zip(sig["reject_reasons"], buy)]
    return sig


def scan_day(cfg: dict, df: pd.DataFrame, model, freshness: dict, date=None, coverage: pd.DataFrame | None = None) -> dict:
    d = pd.Timestamp(date) if date is not None else df["date"].max()
    day = df[df["date"] == d].reset_index(drop=True)
    if day.empty:
        raise ValueError(f"Tidak ada data fitur pada {d.date()}")
    pred = model.predict(day)
    sig = generate_signals(day, pred, cfg)
    sig = apply_freshness_gate(sig, freshness)
    sig = apply_coverage_gate(sig, coverage)          # BUY hanya untuk status cakupan SUCCESS
    sig = apply_concentration_limits(sig, cfg, df[df["date"] <= d])
    ranked = rank_signals(sig, cfg)
    return {"date": d, "signals": sig, "ranked": ranked, "pred": pred}


def _f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def recommendation_record(row: pd.Series, rank: int, cfg: dict, model=None, feats_df=None, driver_feats=None) -> dict:
    h = cfg["labels"]["SWING_HORIZON"]
    pros, cons = build_reasoning(row)
    drivers = model_drivers(model, feats_df.loc[[row.name]], driver_feats) if model is not None and driver_feats else []
    return {
        "rank": rank, "ticker": row["ticker"], "name": row.get("name"), "sector": row.get("sector"),
        "setup": row["setup_type"], "decision": row["decision"], "score": _f(row["final_score"], 1),
        "market_regime": row["market_regime"], "sector_score": _f(row.get("sector_score"), 1),
        "probability_bullish": _f(row["probability_bullish"], 3), "probability_neutral": _f(row["probability_neutral"], 3),
        "probability_bearish": _f(row["probability_bearish"], 3), f"expected_return_{h}d": _f(row.get(f"expected_return_{h}d")),
        "prob_hit_tp": _f(row.get("prob_hit_tp"), 3), "close": _f(row["close"], 2),
        "entry_low": _f(row["entry_low"], 2), "entry_ideal": _f(row["entry_ideal"], 2), "entry_high": _f(row["entry_high"], 2),
        "stop_loss": _f(row["stop_loss"], 2), "stop_method": row["stop_method"],
        "take_profit_1": _f(row["take_profit_1"], 2), "take_profit_2": _f(row["take_profit_2"], 2),
        "risk_reward": _f(row["risk_reward"], 2), "confidence": row["confidence"],
        "position_size": int(row["position_size"]), "lots": int(row["position_lots"]),
        "capital_required": _f(row["capital_required"], 0), "risk_amount": _f(row["risk_amount"], 0),
        "estimated_loss": _f(row["estimated_loss"], 0),
        "holding_period": f"{max(1, h - 2)}-{cfg['strategy']['max_holding_days']} trading days",
        "reasons": pros, "risks": cons, "model_drivers": drivers, "reject_reasons": list(row["reject_reasons"]),
    }


def summarize_scan(cfg: dict, res: dict, model, model_version: str, freshness: dict, synthetic: bool,
                   driver_feats: list[str] | None = None) -> dict:
    sig, ranked = res["signals"], res["ranked"]
    feats_df = sig[model.features] if model is not None else None
    recs = []
    for i, (_, r) in enumerate(ranked.iterrows(), start=1):
        row = sig.loc[sig.index[sig["ticker"] == r["ticker"]][0]]
        recs.append(recommendation_record(row, i, cfg, model, feats_df, driver_feats))
    watch = sig[sig["decision"] == "WATCHLIST"].sort_values("final_score", ascending=False).head(10)
    reg = sig.iloc[0]
    if recs:
        headline = f"TOP {len(recs)} SWING OPPORTUNITIES"
    elif not freshness.get("trading_allowed", True):
        headline = f"NO TRADE — data {freshness['status']} (terakhir {freshness.get('latest_database_date')})"
    else:
        headline = "NO HIGH-CONVICTION SETUP TODAY"
    out = {
        "date": res["date"].strftime("%Y-%m-%d"), "headline": headline, "market_regime": reg["market_regime"],
        "market_context": {"regime_score": _f(reg.get("regime_score"), 3), "ihsg_ret20": _f(reg.get("idx_ret20")),
                           "breadth_above_sma50": _f(reg.get("breadth_above_sma50"), 3),
                           "max_recommendations_today": get(cfg, f"regime.rules.{reg['market_regime']}.max_recommendations")},
        "data_freshness": freshness, "universe_scanned": int(len(sig)), "tradeable": int(sig["is_tradeable"].sum()),
        "decision_counts": {k: int(v) for k, v in sig["decision"].value_counts().items()},
        "recommendations": recs,
        "watchlist": [{"ticker": r["ticker"], "setup": r["setup_type"], "score": _f(r["final_score"], 1),
                       "probability_bullish": _f(r["probability_bullish"], 3), "blocked_by": list(r["reject_reasons"])[:4]}
                      for _, r in watch.iterrows()],
        "model_version": model_version, "data_synthetic": synthetic, "disclaimer": DISCLAIMER,
    }
    if synthetic:
        out["data_warning"] = SYNTHETIC_WARNING
    return out


def analyze_ticker(cfg: dict, res: dict, ticker: str, model, model_version: str, synthetic: bool) -> dict:
    tk = normalize_ticker(ticker)
    sig = res["signals"]
    m = sig["ticker"] == tk
    if not m.any():
        raise ValueError(f"{tk} tidak ada pada {res['date'].date()} (tidak terdaftar / suspensi / belum ada data)")
    row = sig[m].iloc[0]
    rec = recommendation_record(row, 0, cfg, model, sig[model.features], model.features[:15])
    rec.update({"date": res["date"].strftime("%Y-%m-%d"), "model_version": model_version, "data_synthetic": synthetic,
                "context": {k: _f(row.get(k)) for k in ["regime_score", "idx_ret20", "sector_score", "ema20", "ema50", "sma200",
                                                       "adx", "rsi14", "macd_hist", "volume_ratio", "f_rs20", "f_rs_sector20",
                                                       "atr_pct", "sup20", "next_resistance"]}})
    return rec


def _rp(x):
    return "n/a" if x is None else f"${x:,.2f}"


def _pc(x, signed=True):
    return "n/a" if x is None else (f"{x:+.1%}" if signed else f"{x:.0%}")


def sector_label(score):
    if score is None:
        return "n/a"
    return "STRONG" if score >= 65 else "WEAK" if score <= 35 else "NEUTRAL"


def format_analysis(rec: dict) -> str:
    c, h = rec["context"], [k for k in rec if k.startswith("expected_return_")][0]
    rr = rec["risk_reward"]
    lines = ["=" * 48, "AI SWING ANALYSIS", "=" * 48]
    if rec.get("data_synthetic"):
        lines.append("!! " + SYNTHETIC_WARNING)
    lines += [
        f"Ticker            : {rec['ticker']} — {rec.get('name') or ''} ({rec.get('sector') or 'n/a'})",
        f"Tanggal data      : {rec['date']}  | Model: {rec['model_version']}",
        f"Score             : {rec['score']:.0f}/100" if rec["score"] is not None else "Score: n/a",
        f"Market            : {rec['market_regime']} (skor {c['regime_score'] if c['regime_score'] is not None else 'n/a'}, S&P 500 20H {_pc(c['idx_ret20'])})",
        f"Sector            : {sector_label(rec.get('sector_score'))} (sector score {rec.get('sector_score')})",
        f"Trend             : close {_rp(rec['close'])} | EMA20 {_rp(c['ema20'])} | EMA50 {_rp(c['ema50'])} | SMA200 {_rp(c['sma200'])} | ADX {c['adx']:.0f}" if c["adx"] is not None else "Trend: n/a",
        f"Momentum          : RSI14 {c['rsi14']:.0f} | MACD hist {'positif' if (c['macd_hist'] or 0) > 0 else 'negatif'}" if c["rsi14"] is not None else "Momentum: n/a",
        f"Volume            : {c['volume_ratio']:.2f}x rata-rata 20H" if c["volume_ratio"] is not None else "Volume: n/a",
        f"Relative strength : vs S&P 500 {_pc(c['f_rs20'])} | vs sektor {_pc(c['f_rs_sector20'])} (20H)",
        f"Setup             : {rec['setup']}",
        f"Probability       : bullish {_pc(rec['probability_bullish'], False)} | neutral {_pc(rec['probability_neutral'], False)} | bearish {_pc(rec['probability_bearish'], False)}",
        f"Expected return {h.split('_')[-1].upper()}: {_pc(rec[h])}  | P(hit TP) {_pc(rec['prob_hit_tp'], False)}",
        f"Entry             : {_rp(rec['entry_low'])}–{_rp(rec['entry_high'])}  (ideal {_rp(rec['entry_ideal'])})",
        f"Stop loss         : {_rp(rec['stop_loss'])}  ({rec['stop_method']})",
        f"TP1 / TP2         : {_rp(rec['take_profit_1'])} / {_rp(rec['take_profit_2'])}",
        f"Risk/Reward       : 1 : {rr:.2f}" if rr is not None else "Risk/Reward: n/a",
        f"Position          : {rec['position_size']} saham | modal {_rp(rec['capital_required'])} | "
        f"estimasi rugi bila SL {_rp(rec['estimated_loss'])}",
        f"Confidence        : {rec['confidence']}",
        f"Decision          : {rec['decision']}",
        "Reasons:"] + [f"+ {r}" for r in rec["reasons"]] + ["Risks:"] + ([f"- {r}" for r in rec["risks"]] or ["- tidak ada yang menonjol"])
    if rec.get("model_drivers"):
        lines += ["Model drivers:"] + [f"* {r}" for r in rec["model_drivers"]]
    lines.append(DISCLAIMER)
    return "\n".join(lines)
