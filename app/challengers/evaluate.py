"""Evaluasi out-of-sample model challenger dengan definisi yang SAMA seperti model aktif (numpy/pandas saja).

Dipakai ulang dari app/models: `classification_metrics` (roc_auc_bullish, log_loss, brier), `expected_calibration_error`,
`topk_trading_proxy` (top-k per hari, setelah biaya round-trip) dan baris yang dinilai (`y_class` ada & is_tradeable).
Tambahan untuk prakiraan distribusi: IC harian (Spearman lintas emiten), cakupan interval 80%, pinball loss, kualitas σ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import get
from app.models.evaluation import classification_metrics, expected_calibration_error, topk_trading_proxy
from app.models.trainer import round_trip_cost

FORECAST_COLS = ["exp_return", "prob_up", "prob_bullish", "prob_bearish", "vol", "q10", "q50", "q90"]


def to_eval_frame(panel, rows: np.ndarray, fc: dict) -> pd.DataFrame:
    """Gabungkan prakiraan (dict array dari dist.summarize_quantiles) dengan label asli baris `rows`."""
    out = pd.DataFrame({"date": pd.to_datetime(panel.dates[rows]), "ticker": panel.tickers[rows],
                        "is_tradeable": panel.tradeable[rows], "y_class": panel.y_class[rows], "y_ret": panel.y_ret[rows]})
    for k in FORECAST_COLS + ["prob_neutral"]:
        if k in fc:
            out[k] = np.asarray(fc[k], float)
    return out


def daily_ic(df: pd.DataFrame, score: str, target: str = "y_ret", min_names: int = 20) -> pd.Series:
    d = df[["date", score, target]].dropna()
    ranked = d.groupby("date")[[score, target]].rank()
    ranked["date"] = d["date"]
    counts = d.groupby("date").size()
    ok = counts[counts >= min_names].index
    ranked = ranked[ranked["date"].isin(ok)]
    if ranked.empty:                                   # groupby.apply pada frame kosong → DataFrame, bukan Series
        return pd.Series(dtype=float)
    return ranked.groupby("date").apply(lambda g: g[score].corr(g[target]), include_groups=False).dropna()


def _ic_summary(ic: pd.Series, horizon: int) -> dict:
    if len(ic) < 5:
        return {"ic_mean": None, "ic_t": None, "ic_pos_ratio": None, "ic_days": int(len(ic))}
    # tanggal bertumpuk (target H hari) → t-stat dikoreksi kasar dengan √H agar tidak terlalu optimistis
    t = float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic) / max(1, horizon))) if ic.std(ddof=1) > 0 else 0.0
    return {"ic_mean": float(ic.mean()), "ic_t": t, "ic_pos_ratio": float((ic > 0).mean()), "ic_days": int(len(ic))}


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(level * d, (level - 1) * d)))


def evaluate_forecasts(ev: pd.DataFrame, cfg: dict) -> dict:
    """Metrik OOS. `ev` dari `to_eval_frame`. Baris dinilai = y_class ada & is_tradeable (sama dengan model aktif)."""
    h = int(get(cfg, "labels.SWING_HORIZON", 5))
    k = int(get(cfg, "model.comparison_top_k", 5))
    cost = round_trip_cost(cfg)
    p = ev[ev["y_class"].notna() & ev["is_tradeable"].astype(bool)].copy()
    out: dict = {"n": int(len(p)), "eval_start": str(p["date"].min().date()) if len(p) else None,
                 "eval_end": str(p["date"].max().date()) if len(p) else None}
    if len(p) < 50:
        out["status"] = "INSUFFICIENT_DATA"
        return out
    P = p[["prob_bearish", "prob_neutral", "prob_bullish"]].to_numpy(dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    y = p["y_class"].astype(int).to_numpy()
    cm = classification_metrics(y, P)
    out.update({k_: cm[k_] for k_ in ("roc_auc_bullish", "pr_auc_bullish", "log_loss", "brier_bullish", "base_rate_bullish")})
    out["ece_bullish"] = expected_calibration_error((y == 2).astype(int), P[:, 2])
    p["y_ret_h"] = p["y_ret"]
    tk = topk_trading_proxy(p.assign(probability_bullish=P[:, 2]), "probability_bullish", "y_ret_h", k, cost)
    out.update(tk)                                                     # topk_mean_net_return = definisi model aktif
    tk_er = topk_trading_proxy(p, "exp_return", "y_ret_h", k, cost)
    out["topk_by_exp_return_net"] = tk_er["topk_mean_net_return"]
    up = (p["y_ret"] > 0).astype(int)
    if 0 < up.sum() < len(up):
        from sklearn.metrics import roc_auc_score
        out["auc_up"] = float(roc_auc_score(up, p["prob_up"]))
    out.update(_ic_summary(daily_ic(p, "exp_return"), h))
    yl = np.log1p(p["y_ret"].to_numpy(float))
    q10, q50, q90 = (np.log1p(p[c].to_numpy(float)) for c in ("q10", "q50", "q90"))
    out["coverage_80"] = float(np.mean((yl >= q10) & (yl <= q90)))
    out["pinball_mean"] = float(np.mean([_pinball(yl, q10, 0.1), _pinball(yl, q50, 0.5), _pinball(yl, q90, 0.9)]))
    vol_ic = pd.Series(p["vol"].to_numpy()).corr(pd.Series(np.abs(yl)), method="spearman")
    out["vol_rank_corr"] = None if pd.isna(vol_ic) else float(vol_ic)
    out["status"] = "OK"
    return {k_: (None if isinstance(v, float) and not np.isfinite(v) else v) for k_, v in out.items()}


def baseline_metrics(ev: pd.DataFrame, cfg: dict, feature_values: np.ndarray | None) -> dict:
    """Pembanding sederhana pada baris yang SAMA: momentum (fitur baseline), reversal (kebalikannya), rata-rata universe."""
    h = int(get(cfg, "labels.SWING_HORIZON", 5))
    k = int(get(cfg, "model.comparison_top_k", 5))
    cost = round_trip_cost(cfg)
    p = ev[ev["y_class"].notna() & ev["is_tradeable"].astype(bool)].copy()
    if len(p) < 50 or feature_values is None:
        return {}
    p["mom"] = np.asarray(feature_values, float)[ev.index.get_indexer(p.index)]
    p["rev"] = -p["mom"]
    p["y_ret_h"] = p["y_ret"]
    out = {}
    for name in ("mom", "rev"):
        tk = topk_trading_proxy(p, name, "y_ret_h", k, cost)
        ic = _ic_summary(daily_ic(p, name), h)
        out[f"{'momentum' if name == 'mom' else 'reversal'}"] = {"topk_mean_net_return": tk["topk_mean_net_return"],
                                                                 "ic_mean": ic["ic_mean"], "ic_t": ic["ic_t"]}
        out["universe_mean_net_return"] = tk["universe_mean_return"]
    return out


def _f(v, spec: str) -> str:
    """Format aman: None/NaN → 'n/a' (verdict tidak boleh gagal hanya karena satu metrik tidak tersedia)."""
    try:
        if v is None or not np.isfinite(float(v)):
            return "n/a"
        return format(float(v), spec)
    except (TypeError, ValueError):
        return "n/a"


def _ci(b: dict | None, pct: bool = True) -> str:
    if not b or b.get("ci_low") is None:
        return ""
    spec = "+.2%" if pct else "+.3f"
    return f" [CI95 {_f(b['ci_low'], spec)}; {_f(b['ci_high'], spec)}, p={_f(b.get('p_value'), '.3f')}]"


def verdict(metrics: dict, baselines: dict, active: dict | None, sig: dict | None = None) -> list[str]:
    """Kalimat jujur. 'Lebih baik' hanya disebut SIGNIFIKAN bila CI 95% selisih harian berpasangan tidak memuat 0."""
    if metrics.get("status") != "OK":
        return ["Data evaluasi tidak cukup."]
    sig = sig or {}
    msgs = []
    auc = metrics.get("roc_auc_bullish")
    if auc is not None:
        msgs.append(f"AUC bullish {_f(auc, '.3f')} ({'di atas' if auc > 0.52 else 'TIDAK di atas'} ambang promosi 0,52); "
                    f"AUC arah naik {_f(metrics.get('auc_up'), '.3f')}")
    ece, ece_raw = metrics.get("ece_bullish"), metrics.get("ece_bullish_raw")
    if ece is not None:
        msgs.append(f"ECE {_f(ece, '.3f')}" + (f" (sebelum kalibrasi {_f(ece_raw, '.3f')})" if ece_raw is not None and ece_raw != ece else "")
                    + f" → {'lolos' if ece <= 0.06 else 'TIDAK lolos'} syarat ≤ 0,06")
    rule = metrics.get("ranking_rule", "prob_bullish")
    tk_rule = metrics.get("topk_by_rule_net", metrics.get("topk_mean_net_return"))
    if tk_rule is not None:
        msgs.append(f"Aturan ranking (dipilih di validasi): {rule} → top-k net {_f(tk_rule, '+.2%')}{_ci(sig.get('topk'))}")
    if sig.get("ic") and sig["ic"].get("mean") is not None:
        msgs.append(f"IC harian {_f(sig['ic']['mean'], '+.3f')}{_ci(sig['ic'], pct=False)}")
    for name in ("momentum", "reversal"):
        d = sig.get(f"topk_vs_{name}")
        b = (baselines or {}).get(name, {})
        if d and d.get("ci_low") is not None:
            if d["ci_low"] > 0:
                word = "lebih baik secara SIGNIFIKAN"
            elif d["ci_high"] < 0:
                word = "lebih BURUK secara signifikan"
            else:
                word = "perbedaan TIDAK signifikan"
            msgs.append(f"vs baseline {name} ({_f(b.get('topk_mean_net_return'), '+.2%')}): selisih harian "
                        f"{_f(d.get('mean'), '+.2%')}{_ci(d)} → {word}")
        elif b.get("topk_mean_net_return") is not None and tk_rule is not None:
            msgs.append(f"vs baseline {name}: {_f(tk_rule, '+.2%')} vs {_f(b['topk_mean_net_return'], '+.2%')} "
                        "(signifikansi tidak tersedia)")
    if active:
        a_tk, a_auc = active.get("topk_mean_net_return"), active.get("roc_auc_bullish")
        tk_pb = metrics.get("topk_mean_net_return")
        if a_tk is not None and tk_pb is not None:
            auc_txt = f", AUC {_f(auc, '.3f')} vs {_f(a_auc, '.3f')}" if auc is not None and a_auc is not None else ""
            msgs.append(f"vs model aktif ({active.get('version')}), definisi sama (top-k P(bullish)): {_f(tk_pb, '+.2%')} vs "
                        f"{_f(a_tk, '+.2%')}{auc_txt} → {active.get('comparison', 'tidak bisa dinilai')} "
                        f"(perbandingan kasar: prediksi test model aktif tidak tersedia untuk uji berpasangan)")
    t = sig.get("topk") or {}
    if t.get("ci_low") is not None and t["ci_low"] > 0:
        msgs.append("Top-k net > 0 secara signifikan setelah biaya pada periode test (satu periode — belum bukti cukup).")
    elif tk_rule is not None:
        msgs.append("Top-k net setelah biaya TIDAK terbukti > 0 secara signifikan: belum layak untuk keputusan transaksi.")
    return msgs


# ============================================================================== signifikansi (bootstrap blok per tanggal)
def daily_topk(df: pd.DataFrame, score: str, ret: str, k: int, cost: float) -> pd.Series:
    """Rata-rata return bersih top-k per tanggal (seri harian; dasar `topk_mean_net_return`)."""
    d = df[["date", score, ret]].dropna()
    top = d.sort_values(score, ascending=False).groupby("date").head(k)
    return (top[ret] - cost).groupby(top["date"]).mean().sort_index()


def block_bootstrap(series, block: int = 5, n_boot: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap atas seri harian. Blok = horizon label, karena return H hari dari tanggal berdekatan
    saling bertumpuk (resampling per tanggal tunggal akan terlalu optimistis).
    Return: rata-rata, CI 95%, p-value dua sisi untuk H0: rata-rata = 0."""
    x = np.asarray(pd.Series(series).dropna(), float)
    n = len(x)
    if n < max(10, 2 * block):
        return {"mean": float(x.mean()) if n else None, "ci_low": None, "ci_high": None, "p_value": None, "n_days": int(n)}
    rng = np.random.default_rng(seed)
    b = max(1, min(block, n // 2))
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    centered = means - x.mean()                                         # distribusi di bawah H0 (digeser ke 0)
    p = float(np.mean(np.abs(centered) >= abs(x.mean())))
    return {"mean": float(x.mean()), "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
            "p_value": max(p, 1.0 / n_boot), "n_days": int(n)}


def significance(ev: pd.DataFrame, cfg: dict, score: str, baseline_values: np.ndarray | None = None,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Uji untuk top-k net (skor `score`) dan IC harian: terhadap nol dan PASANGAN harian vs baseline pada baris sama."""
    h = int(get(cfg, "labels.SWING_HORIZON", 5))
    k = int(get(cfg, "model.comparison_top_k", 5))
    cost = round_trip_cost(cfg)
    p = ev[ev["y_class"].notna() & ev["is_tradeable"].astype(bool)].copy()
    if len(p) < 50:
        return {}
    model_topk = daily_topk(p, score, "y_ret", k, cost)
    out = {"block_days": h, "n_boot": n_boot,
           "topk": block_bootstrap(model_topk, h, n_boot, seed),
           "ic": block_bootstrap(daily_ic(p, "exp_return"), h, n_boot, seed)}
    if baseline_values is not None:
        p["mom"] = np.asarray(baseline_values, float)[ev.index.get_indexer(p.index)]
        p["rev"] = -p["mom"]
        for name, col in (("momentum", "mom"), ("reversal", "rev")):
            diff = (model_topk - daily_topk(p, col, "y_ret", k, cost)).dropna()
            out[f"topk_vs_{name}"] = block_bootstrap(diff, h, n_boot, seed)
    return out


def compare_to_value(boot: dict, value) -> str:
    """Bandingkan CI challenger dengan angka TITIK model aktif (prediksi test model aktif tidak tersedia untuk uji
    berpasangan — perbandingan ini kasar)."""
    if not boot or boot.get("ci_low") is None or value is None:
        return "tidak bisa dinilai"
    if value > boot["ci_high"]:
        return "model aktif lebih tinggi dari CI 95% challenger"
    if value < boot["ci_low"]:
        return "model aktif lebih rendah dari CI 95% challenger"
    return "di dalam CI 95% challenger — perbedaan tidak signifikan"


# ============================================================================== aturan ranking (dipilih di fold validasi)
RANK_RULES = ("prob_bullish", "exp_return", "risk_adjusted", "prob_up")


def rank_score(fc: dict | pd.DataFrame, rule: str) -> np.ndarray:
    """Skor untuk mengurutkan saham. risk_adjusted = ekspektasi return / σ (tidak bias ke saham paling volatil)."""
    get_ = (lambda c: np.asarray(fc[c], float))
    if rule == "risk_adjusted":
        return get_("exp_return") / np.maximum(get_("vol"), 1e-6)
    if rule not in RANK_RULES:
        raise ValueError(f"aturan ranking tidak dikenal: {rule}")
    return get_(rule)


def choose_ranking(ev_val: pd.DataFrame, cfg: dict) -> tuple[str, dict]:
    """Pilih aturan dengan top-k net tertinggi di FOLD VALIDASI (test tidak dilihat). Seri: urutan RANK_RULES."""
    k = int(get(cfg, "model.comparison_top_k", 5))
    cost = round_trip_cost(cfg)
    p = ev_val[ev_val["y_class"].notna() & ev_val["is_tradeable"].astype(bool)].copy()
    scores = {}
    for rule in RANK_RULES:
        p["_s"] = rank_score(p, rule)
        s = daily_topk(p, "_s", "y_ret", k, cost)
        scores[rule] = float(s.mean()) if len(s) else float("-inf")
    best = max(RANK_RULES, key=lambda r: (scores[r], -RANK_RULES.index(r)))
    return best, scores
