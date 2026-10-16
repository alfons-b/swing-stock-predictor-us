"""Statistik yang dipakai quant untuk menilai apakah hasil backtest NYATA atau kebetulan.

Mengapa perlu (temuan audit 16 Okt 2026): run test produksi memberi Sharpe 0,47 dari 85 trade dalam 6 bulan.
Galat baku Sharpe tahunan dari 6 bulan data harian ≈ √(1/0,5) ≈ 1,4 — artinya angka 0,47 tidak bisa dibedakan
dari nol, dan juga tidak bisa dibedakan dari 2. Membandingkan langsung dengan S&P 500 buy & hold juga menyesatkan:
strategi rata-rata hanya 43% terinvestasi, jadi pembanding yang adil adalah eksposur pasar yang sama (beta).

Isi:
- alpha_beta     : regresi return harian strategi terhadap indeks → beta, alpha tahunan, t-stat alpha,
                   information ratio, korelasi. Alpha > 0 yang signifikan = keunggulan di luar eksposur pasar.
- sharpe_ci      : interval kepercayaan Sharpe dengan stationary block bootstrap (autokorelasi dipertahankan).
- probabilistic_sharpe / deflated_sharpe : Bailey & López de Prado (2012, 2014). PSR = P(Sharpe sebenarnya > acuan)
                   memperhitungkan panjang sampel, skew, kurtosis. DSR menaikkan acuan sesuai jumlah konfigurasi yang
                   dicoba (multiple testing) — makin banyak variasi yang dicoba, makin tinggi bukti yang dibutuhkan.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Invers CDF normal (Acklam). Cukup akurat untuk ambang statistik (galat < 1e-9)."""
    if not 0 < p < 1:
        return float("inf") if p >= 1 else float("-inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def daily_returns(equity: pd.DataFrame) -> pd.Series:
    eq = equity.set_index("date")["equity"].astype(float)
    return eq.pct_change().dropna()


def sharpe(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    s = r.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * r.mean() / s) if len(r) > 2 and s > 0 else float("nan")


def alpha_beta(strategy: pd.Series, index: pd.Series) -> dict:
    """strategy, index: return harian berindeks tanggal. OLS r_s = α + β r_m + ε."""
    df = pd.concat({"s": pd.Series(strategy), "m": pd.Series(index)}, axis=1).dropna()
    if len(df) < 20 or df["m"].var() <= 0:
        return {}
    x, y = df["m"].to_numpy(), df["s"].to_numpy()
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha_d = float(y.mean() - beta * x.mean())
    resid = y - (alpha_d + beta * x)
    n = len(df)
    se_alpha = float(np.sqrt(np.sum(resid ** 2) / (n - 2)) * np.sqrt(1 / n + x.mean() ** 2 / np.sum((x - x.mean()) ** 2)))
    te = float(np.std(resid, ddof=1))
    return {"beta": round(beta, 3), "alpha_annual": round(alpha_d * TRADING_DAYS, 4),
            "alpha_tstat": round(alpha_d / se_alpha, 2) if se_alpha > 0 else None,
            "information_ratio": round(float(np.sqrt(TRADING_DAYS) * alpha_d / te), 2) if te > 0 else None,
            "correlation": round(float(np.corrcoef(x, y)[0, 1]), 3), "n_days": n,
            # return yang SEHARUSNYA didapat hanya dari eksposur pasar (beta × indeks), tanpa skill
            "beta_matched_index_return": round(float(np.prod(1 + beta * x) - 1), 4)}


def _stationary_bootstrap_idx(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    p = 1.0 / max(mean_block, 1.0)
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
    return idx


def sharpe_ci(r: pd.Series, n_boot: int = 1000, alpha: float = 0.10, mean_block: float = 5.0, seed: int = 7) -> dict:
    """Interval kepercayaan (1−alpha) Sharpe tahunan via stationary bootstrap (Politis & Romano 1994)."""
    r = pd.Series(r).dropna().to_numpy()
    if len(r) < 30:
        return {}
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_boot):
        s = r[_stationary_bootstrap_idx(len(r), mean_block, rng)]
        sd = s.std(ddof=1)
        if sd > 0:
            sims.append(np.sqrt(TRADING_DAYS) * s.mean() / sd)
    if not sims:
        return {}
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    return {"sharpe_ci_low": round(float(lo), 2), "sharpe_ci_high": round(float(hi), 2),
            "ci_level": 1 - alpha, "p_sharpe_le_0": round(float(np.mean(np.asarray(sims) <= 0)), 3)}


def probabilistic_sharpe(r: pd.Series, sr_benchmark_annual: float = 0.0) -> float | None:
    """PSR: P(SR sebenarnya > acuan), memperhitungkan panjang sampel, skew dan kurtosis (Bailey & López de Prado)."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 30 or r.std(ddof=1) <= 0:
        return None
    sr = r.mean() / r.std(ddof=1)                      # per hari
    sr_b = sr_benchmark_annual / np.sqrt(TRADING_DAYS)
    skew, kurt = float(r.skew()), float(r.kurt()) + 3.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return round(_norm_cdf((sr - sr_b) * math.sqrt(n - 1) / denom), 4)


def expected_max_sharpe(n_trials: int, var_trials_daily: float) -> float:
    """E[max Sharpe] dari n percobaan independen bila semua sebenarnya tanpa skill (acuan DSR, per hari)."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(max(var_trials_daily, 0.0)) * ((1 - _EULER) * _norm_ppf(1 - 1 / n_trials)
                                                    + _EULER * _norm_ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(r: pd.Series, n_trials: int, var_trials_daily: float | None = None) -> float | None:
    """DSR: PSR dengan acuan = Sharpe maksimum yang diharapkan dari keberuntungan setelah n_trials konfigurasi.
    var_trials_daily default = varians Sharpe per hari di bawah nol-skill ≈ 1/(n−1)."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 30:
        return None
    v = var_trials_daily if var_trials_daily is not None else 1.0 / (n - 1)
    sr0_daily = expected_max_sharpe(max(1, int(n_trials)), v)
    return probabilistic_sharpe(r, sr0_daily * np.sqrt(TRADING_DAYS))


def significance_report(equity: pd.DataFrame, index_close: pd.Series | None, n_trials: int = 1) -> dict:
    """Ringkasan lengkap untuk satu run backtest (dipakai runner & dashboard)."""
    r = daily_returns(equity)
    out = {"n_days": int(len(r))}
    out.update(sharpe_ci(r))
    psr = probabilistic_sharpe(r)
    dsr = deflated_sharpe(r, n_trials)
    out["psr_vs_0"] = psr
    out["dsr"] = dsr
    out["n_trials_assumed"] = int(n_trials)
    if index_close is not None and len(index_close):
        ix = pd.Series(index_close).astype(float).sort_index().pct_change().dropna()
        out.update(alpha_beta(r, ix))
    out["verdict"] = verdict(out)
    return out


def verdict(s: dict) -> str:
    """Kalimat jujur: bukti apa yang benar-benar dimiliki run ini."""
    dsr, t = s.get("dsr"), s.get("alpha_tstat")
    lo, hi = s.get("sharpe_ci_low"), s.get("sharpe_ci_high")
    ci = f"Sharpe {int(100 * s.get('ci_level', 0.9))}% CI [{lo}, {hi}]" if lo is not None else "CI Sharpe n/a"
    alpha = (f"alpha {s['alpha_annual']:+.1%}/th (t={t}), beta {s['beta']}" if s.get("alpha_annual") is not None
             else "alpha n/a")
    if dsr is not None and dsr >= 0.95 and t is not None and t >= 2:
        tag = "BUKTI KUAT"
    elif (dsr is not None and dsr >= 0.80) or (t is not None and t >= 1.5):
        tag = "BUKTI LEMAH"
    else:
        tag = "BELUM ADA BUKTI EDGE"
    return f"{tag}: {ci}; {alpha}; DSR {dsr if dsr is not None else 'n/a'}"
