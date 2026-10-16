"""Backtest jobs (§17, §41).

walk_forward : model dilatih ulang per fold (OOS), pipeline sinyal identik dengan scan, biaya + slippage,
               board lot, position sizing. Pembanding: strategi teknikal-saja & S&P 500 buy-and-hold.
live         : mensimulasikan sinyal yang BENAR-BENAR dikeluarkan sistem (tabel trading_signals) —
               satu-satunya bukti yang sepenuhnya out-of-sample.
"""
from __future__ import annotations

import copy
import uuid

import pandas as pd

from app.backtest.engine import BacktestEngine, benchmark_buy_hold
from app.backtest.walk_forward import make_folds
from app.config import config_hash
from app.models.trainer import walk_forward_predict
from app.risk.concentration import apply_concentration_limits
from app.scanner.ranking import generate_signals, rank_signals
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def prediction_columns(pred: pd.DataFrame) -> list[str]:
    return [c for c in pred.columns if c.startswith("prob") or c.startswith("expected_") or c == "base_rate_bullish"]


def technical_only_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["scoring"]["final_weights"] = {"technical": 1.0, "ml": 0.0, "fundamental": 0.0, "sentiment": 0.0}
    c["strategy"]["MIN_CONFIDENCE"] = 0.0
    c["strategy"]["cost_multiple"] = 0.0             # tanpa model tidak ada expected return → gerbang biaya dimatikan
    c["strategy"]["min_net_expected_return"] = -1.0
    for r in c["regime"]["rules"].values():
        r["prob_add"] = 0.0
    return c


def neutral_predictions(index, horizon: int) -> pd.DataFrame:
    return pd.DataFrame({"probability_bearish": 0.25, "probability_neutral": 0.40, "probability_bullish": 0.35,
                         "prob_dispersion": 0.0, f"expected_return_{horizon}d": 0.001}, index=index)


def n_trials(cfg: dict) -> int:
    """Perkiraan jumlah konfigurasi yang dicoba sebelum memilih yang dilaporkan (untuk Deflated Sharpe):
    kandidat model × kombinasi grid tuning. Bisa dioverride `backtest.n_trials`."""
    from app.config import get
    if get(cfg, "backtest.n_trials"):
        return int(get(cfg, "backtest.n_trials"))
    grid = 1
    for v in (get(cfg, "tuning", {}) or {}).values():
        if isinstance(v, list) and v:
            grid *= len(v)
    return max(1, len(get(cfg, "model.candidates", []) or [1]) * grid)


def _significance(cfg, res, index: pd.DataFrame) -> dict:
    from app.backtest.stats import significance_report
    eq = res.get("equity")
    if eq is None or len(eq) < 30:
        return {}
    ix = index.set_index("date")["close"]
    ix = ix[(ix.index >= eq["date"].min()) & (ix.index <= eq["date"].max())]
    return significance_report(eq, ix, n_trials(cfg))


def _run_period(cfg, df, pred, start, end) -> dict:
    pcols = prediction_columns(pred)
    feat = df.merge(pred[["date", "ticker"] + pcols], on=["date", "ticker"], how="inner")
    ranked = rank_signals(apply_concentration_limits(generate_signals(feat.drop(columns=pcols), feat[pcols], cfg), cfg, df), cfg)
    res = BacktestEngine(cfg).run(ranked, df, start, end)
    tcfg = technical_only_cfg(cfg)
    base = feat.drop(columns=pcols)
    r_t = rank_signals(apply_concentration_limits(
        generate_signals(base, neutral_predictions(base.index, cfg["labels"]["SWING_HORIZON"]), tcfg), tcfg, df), tcfg)
    res_t = BacktestEngine(tcfg).run(r_t, df, start, end)
    index = df[["date", "idx_close"]].drop_duplicates("date").rename(columns={"idx_close": "close"})
    res["metrics"]["significance"] = _significance(cfg, res, index)
    res_t["metrics"]["significance"] = _significance(cfg, res_t, index)
    return {"strategy": res["metrics"], "technical_only": res_t["metrics"],
            "ihsg_buy_hold": benchmark_buy_hold(index, start, end, float(cfg["backtest"]["initial_capital"])),
            "trades": res["trades"], "no_trade_days_pct": float(1 - ranked["date"].nunique() / max(1, feat["date"].nunique()))}


def walk_forward_backtest(cfg: dict, df: pd.DataFrame, classifiers: list[str], features: list[str], repo,
                          model_version: str, last_n_folds: int | None = None) -> dict:
    folds = make_folds(df["date"], cfg)
    val = [f for f in folds if f.kind == "validation"]
    if last_n_folds:
        val = val[-int(last_n_folds):]
    c = copy.deepcopy(cfg)
    out = {}
    for kind, sel in (("validation", val), ("test", [f for f in folds if f.kind == "test"])):
        if not sel:
            continue
        pred, _ = walk_forward_predict(df, c, classifiers, features=features, folds=sel)
        start, end = pred["date"].min(), pred["date"].max()
        r = _run_period(cfg, df, pred, start, end)
        run_id = f"wf-{kind}-{pd.Timestamp.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
        repo.save_backtest(run_id, f"walk_forward_{kind}", r["strategy"],
                           {"technical_only": r["technical_only"], "ihsg_buy_hold": r["ihsg_buy_hold"],
                            "no_trade_days_pct": r["no_trade_days_pct"]}, r["trades"], model_version, config_hash(cfg))
        r["run_id"] = run_id
        out[kind] = {k: v for k, v in r.items() if k != "trades"}
        log.info("Backtest %s %s..%s: return %+.1f%% | S&P 500 %+.1f%% | trades %d | %s", kind, start.date(), end.date(),
                 100 * r["strategy"].get("total_return", 0), 100 * r["ihsg_buy_hold"].get("total_return", 0),
                 r["strategy"].get("number_of_trades", 0), r["strategy"].get("significance", {}).get("verdict", ""))
    return out


def live_backtest(cfg: dict, repo, df: pd.DataFrame) -> dict | None:
    sig = repo.db.query_df("SELECT signal_date AS date, ticker, setup AS setup_type, entry_low, entry_ideal, entry_high, "
                           "stop_loss, tp1 AS take_profit_1, tp2 AS take_profit_2, risk_reward, model_version "
                           "FROM trading_signals WHERE decision = 'BUY'", parse_dates=["date"])
    if sig.empty:
        return None
    ctx = df[["date", "ticker", "market_regime", "avg_value20"]]
    sig = sig.merge(ctx, on=["date", "ticker"], how="left")
    sig["market_regime"] = sig["market_regime"].fillna("NEUTRAL")
    start, end = sig["date"].min(), df["date"].max()
    res = BacktestEngine(cfg).run(sig, df, start, end)
    index = df[["date", "idx_close"]].drop_duplicates("date").rename(columns={"idx_close": "close"})
    bench = benchmark_buy_hold(index, start, end, float(cfg["backtest"]["initial_capital"]))
    run_id = f"live-{pd.Timestamp.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    repo.save_backtest(run_id, "live_signals", res["metrics"], {"ihsg_buy_hold": bench}, res["trades"],
                       ",".join(sorted(sig["model_version"].dropna().unique())), config_hash(cfg))
    return {"run_id": run_id, "strategy": res["metrics"], "ihsg_buy_hold": bench}
