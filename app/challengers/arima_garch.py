"""ARIMA-GARCH per emiten.

Model (per emiten, atas log return harian r_t dalam persen):
    mean      : ARIMA(p,0,q) pada r_t  (= ARIMA(p,1,q) pada log harga), konstanta = rata-rata μ
                (r_t − μ) = Σ φ_i (r_{t−i} − μ) + ε_t + Σ θ_j ε_{t−j}
    variance  : GARCH(1,1), inovasi Student-t:  σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}
Estimasi dua tahap (praktik umum): ARMA dengan statsmodels (orde dipilih AIC dari grid kecil), lalu GARCH dengan `arch`
pada residual ARMA. Library hanya dipakai untuk ESTIMASI parameter.

Prakiraan H hari untuk setiap tanggal t dihitung dengan rekursi numpy memakai parameter TETAP (scipy.signal.lfilter):
    mean H hari = Σ_{k=1..H} E[r_{t+k} | F_t]
    var  H hari = Σ_{k=1..H} σ²_{t+k|t} · (Σ_{j=0..H−k} ψ_j)²     (ψ = bobot MA(∞) dari ARMA)
Distribusi jumlah H hari didekati normal (jumlah inovasi t mendekati normal). Evaluasi: parameter diestimasi hanya dari
data ≤ akhir training (embargo), lalu dipakai tetap untuk setiap tanggal di periode test → tanpa look-ahead.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from scipy.signal import lfilter

SCALE = 100.0                     # log return × 100 (stabilitas numerik estimasi)


# ============================================================================== rekursi numpy (tanpa library berat)
def arma_residuals(r: np.ndarray, mu: float, ar, ma) -> np.ndarray:
    """ε_t dari (1 − Σφ L^i)(r − μ) = (1 + Σθ L^j) ε  (inisialisasi ε = 0 sebelum awal sampel)."""
    b = np.r_[1.0, -np.asarray(ar, float)]
    a = np.r_[1.0, np.asarray(ma, float)]
    return lfilter(b, a, np.asarray(r, float) - mu)


def psi_weights(ar, ma, n: int) -> np.ndarray:
    ar, ma = np.asarray(ar, float), np.asarray(ma, float)
    psi = np.zeros(n)
    psi[0] = 1.0
    for j in range(1, n):
        v = ma[j - 1] if j - 1 < len(ma) else 0.0
        for i in range(1, min(j, len(ar)) + 1):
            v += ar[i - 1] * psi[j - i]
        psi[j] = v
    return psi


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    """x_{t−k} (NaN → 0 di awal sampel, konsisten dengan inisialisasi filter)."""
    if k == 0:
        return x
    out = np.zeros_like(x)
    out[k:] = x[:-k]
    return out


def arma_mean_paths(r: np.ndarray, eps: np.ndarray, mu: float, ar, ma, h: int) -> np.ndarray:
    """(T, h): E[r_{t+k} | F_t] untuk setiap t (vektor), k = 1..h."""
    ar, ma = list(ar), list(ma)
    w = np.asarray(r, float) - mu
    preds = {}                                   # k → array w_{t+k|t}

    def w_at(off):                               # w_{t+off|t}
        return preds[off] if off >= 1 else _lag(w, -off)

    def e_at(off):                               # ε_{t+off} (0 untuk masa depan)
        return np.zeros_like(w) if off >= 1 else _lag(eps, -off)

    for k in range(1, h + 1):
        v = np.zeros_like(w)
        for i, phi in enumerate(ar, start=1):
            v += phi * w_at(k - i)
        for j, th in enumerate(ma, start=1):
            v += th * e_at(k - j)
        preds[k] = v
    return np.column_stack([preds[k] + mu for k in range(1, h + 1)])


def garch_variance(eps: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    """σ²_t (varians bersyarat ε_t diketahui F_{t−1}) untuk setiap t."""
    e2_prev = _lag(np.asarray(eps, float) ** 2, 1)
    e2_prev[0] = s0
    u = omega + alpha * e2_prev
    s, _ = lfilter([1.0], [1.0, -beta], u, zi=[beta * s0])
    return s


def garch_forecast_paths(eps: np.ndarray, sig2: np.ndarray, omega, alpha, beta, h: int) -> np.ndarray:
    """(T, h): σ²_{t+k|t}, k = 1..h."""
    out = np.empty((len(eps), h))
    out[:, 0] = omega + alpha * np.asarray(eps) ** 2 + beta * sig2
    for k in range(1, h):
        out[:, k] = omega + (alpha + beta) * out[:, k - 1]
    return out


def forecast_series(r_log: np.ndarray, params: dict, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Untuk seluruh deret log return (satuan asli): (μ_H, σ_H) log return kumulatif H hari di setiap t."""
    r = np.nan_to_num(np.asarray(r_log, float) * SCALE)
    mu, ar, ma = params["mu"], params["ar"], params["ma"]
    eps = arma_residuals(r, mu, ar, ma)
    mean_paths = arma_mean_paths(r, eps, mu, ar, ma, h)
    if params.get("fallback") == "ewma":
        lam = params.get("lambda", 0.94)
        s2 = lfilter([1 - lam], [1, -lam], _lag(eps ** 2, 1), zi=[lam * params["s0"]])[0]
        s2_next = np.maximum((1 - lam) * eps ** 2 + lam * s2, 1e-8)   # σ²_{t+1|t}; EWMA → datar ke depan
        var_paths = np.repeat(s2_next[:, None], h, axis=1)
    else:
        s2 = garch_variance(eps, params["omega"], params["alpha"], params["beta"], params["s0"])
        var_paths = garch_forecast_paths(eps, s2, params["omega"], params["alpha"], params["beta"], h)
    psi = psi_weights(ar, ma, h)
    cum = np.array([psi[:h - k].sum() for k in range(h)])            # Σ_{j=0..h−k} ψ_j untuk k = 1..h
    var_h = (var_paths * cum[None, :] ** 2).sum(axis=1)
    return mean_paths.sum(axis=1) / SCALE, np.sqrt(np.maximum(var_h, 1e-10)) / SCALE


# ============================================================================== estimasi (statsmodels + arch)
def fit_one(r_log: np.ndarray, orders, garch_cfg: dict, min_obs: int = 250) -> dict:
    """Estimasi parameter untuk satu emiten. Tidak pernah melempar error: gagal → fallback yang ditandai."""
    r = np.asarray(r_log, float)
    r = r[np.isfinite(r)] * SCALE
    if len(r) < min_obs:
        return {"status": "INSUFFICIENT_DATA", "n": int(len(r))}
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from statsmodels.tsa.arima.model import ARIMA
        for p, q in orders:
            try:
                res = ARIMA(r, order=(int(p), 0, int(q)), trend="c").fit()
                if not np.isfinite(res.aic):
                    continue
                if best is None or res.aic < best[0]:
                    pn = dict(zip(res.model.param_names, res.params))
                    best = (float(res.aic), int(p), int(q), float(pn.get("const", r.mean())),
                            [float(pn[f"ar.L{i}"]) for i in range(1, int(p) + 1)],
                            [float(pn[f"ma.L{j}"]) for j in range(1, int(q) + 1)])
            except Exception:
                continue
    if best is None:
        best = (None, 0, 0, float(r.mean()), [], [])
    aic, p, q, mu, ar, ma = best
    eps = arma_residuals(r, mu, ar, ma)
    s0 = float(np.var(eps))
    params = {"status": "OK", "n": int(len(r)), "p": p, "q": q, "aic": aic, "mu": mu, "ar": ar, "ma": ma, "s0": s0}
    try:
        from arch import arch_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = arch_model(eps, mean="Zero", vol="GARCH", p=int(garch_cfg.get("p", 1)), q=int(garch_cfg.get("q", 1)),
                           dist=str(garch_cfg.get("dist", "t")), rescale=False).fit(disp="off", show_warning=False)
        gp = g.params
        omega, alpha, beta = float(gp["omega"]), float(gp["alpha[1]"]), float(gp["beta[1]"])
        if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 0.999):
            raise ValueError("GARCH tidak stasioner")
        params.update({"omega": omega, "alpha": alpha, "beta": beta, "nu": float(gp.get("nu", np.nan)),
                       "s0": float(omega / (1 - alpha - beta))})
    except Exception as e:
        params.update({"fallback": "ewma", "lambda": 0.94, "garch_error": type(e).__name__})
    return params


def fit_all(returns: dict, orders, garch_cfg: dict, min_obs: int, n_jobs: int = -1) -> dict:
    """{ticker: deret log return} → {ticker: params}. Paralel per emiten (joblib)."""
    from joblib import Parallel, delayed
    items = list(returns.items())
    res = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(fit_one)(r, orders, garch_cfg, min_obs) for _, r in items)
    return {t: p for (t, _), p in zip(items, res)}


def to_artifact(params: dict, meta: dict) -> bytes:
    return json.dumps({"meta": meta, "params": params}, separators=(",", ":")).encode("utf-8")


def from_artifact(blob: bytes) -> tuple[dict, dict]:
    d = json.loads(bytes(blob).decode("utf-8"))
    return d["params"], d["meta"]
