"""Estimasi nilai wajar per saham sebagai RENTANG (low / base / high), per metode.

Setiap fungsi mengembalikan dict:
    {"method", "valid", "reason", "low", "base", "high", "assumptions", "detail"}
Nilai dalam mata uang HARGA (USD) per saham. Metode yang tidak layak → valid=False + alasan (bukan nilai nol).

Metode:
- relative    : kuartil 25/50/75 multiple peer × metrik per saham emiten (PER, PBV, EV/EBITDA bila relevan).
- historical  : persentil 25/50/75 multiple emiten sendiri (bulanan, ≥ 24 bulan) × metrik per saham terkini.
- dcf         : FCFF 5 tahun + nilai terminal Gordon; sensitivitas discount rate × terminal growth;
                skenario low/base/high. Tidak untuk bank/lembaga keuangan. Komoditas: FCF dinormalisasi
                (rata-rata siklus) tanpa pertumbuhan eksplisit.
- dividend    : Gordon growth atas dividen per saham (hanya bila dividen konsisten & payout ≤ 100%).
- justified_pb: bank/lembaga keuangan: P/B wajar = (ROE − g) / (COE − g).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.valuation.valuation_metrics import FINANCIAL_TYPES


def _res(method, valid=False, reason="", low=None, base=None, high=None, assumptions=None, detail=None):
    return {"method": method, "valid": bool(valid), "reason": reason, "low": low, "base": base, "high": high,
            "assumptions": assumptions or {}, "detail": detail or {}}


def cost_of_equity(vcfg: dict, beta: float | None = None) -> float:
    b = beta if beta is not None and 0.3 <= beta <= 2.5 else float(vcfg.get("default_beta", 1.0))
    return float(vcfg.get("risk_free_rate", 0.068)) + b * float(vcfg.get("equity_risk_premium", 0.075))


# ------------------------------------------------------------------------------------------------- relative
def relative_value(items: dict, stype: str, shares: float, fx: float, peers: dict) -> dict:
    """peers: {metric: {"q25","q50","q75","n","level","group"}} untuk multiple yang tersedia di kelompok emiten."""
    if not shares or shares <= 0 or fx is None:
        return _res("relative", reason="jumlah saham / kurs tidak tersedia")
    cv = lambda v: None if v is None else v * fx  # noqa: E731
    per_share = {}
    ni, eq = cv(items.get("net_income")), cv(items.get("total_equity"))
    if ni is not None and ni > 0:
        per_share["per"] = ni / shares
    if eq is not None and eq > 0:
        per_share["pbv"] = eq / shares
    net_debt = None
    if stype not in FINANCIAL_TYPES and items.get("total_debt") is not None and items.get("cash") is not None:
        net_debt = cv(items["total_debt"]) - cv(items["cash"])
        ebitda = cv(items.get("ebitda"))
        if ebitda is not None and ebitda > 0:
            per_share["ev_ebitda"] = ebitda
    if stype in FINANCIAL_TYPES:
        per_share.pop("ev_ebitda", None)
    lows, bases, highs, used = [], [], [], {}
    for metric, base_val in per_share.items():
        q = peers.get(metric)
        if not q:
            continue
        if metric == "ev_ebitda":
            vals = [(q[k] * base_val - net_debt) / shares for k in ("q25", "q50", "q75")]
        else:
            vals = [q[k] * base_val for k in ("q25", "q50", "q75")]
        if vals[1] <= 0:
            continue
        lows.append(max(vals[0], 0.0))
        bases.append(vals[1])
        highs.append(vals[2])
        used[metric] = {"peer_q25": q["q25"], "peer_median": q["q50"], "peer_q75": q["q75"], "n_peers": q["n"],
                        "peer_group": f"{q.get('level')}={q.get('group')}", "value_base": vals[1]}
    if not bases:
        return _res("relative", reason="tidak ada multiple valid dengan kelompok peer ≥ min_peers")
    return _res("relative", True, low=float(np.median(lows)), base=float(np.median(bases)), high=float(np.median(highs)),
                detail=used)


# ----------------------------------------------------------------------------------------------- historical
def historical_multiples(prices: pd.DataFrame, statements: pd.DataFrame, shares: float, fx: float,
                         years: int = 5) -> pd.DataFrame:
    """Multiple bulanan emiten sendiri, point-in-time: tiap akhir bulan memakai laporan FY yang SUDAH diketahui.

    Harga Yahoo sudah disesuaikan split → per-saham memakai jumlah saham TERKINI agar konsisten
    (keterbatasan: penerbitan saham baru/rights issue membuat histori sedikit bias — dicatat di laporan).
    """
    if prices is None or prices.empty or statements is None or statements.empty or not shares or fx is None:
        return pd.DataFrame()
    p = prices[["date", "close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] >= p["date"].max() - pd.DateOffset(years=years)]
    m = p.set_index("date")["close"].resample("ME").last().dropna().reset_index()
    fy = statements[statements["period_type"] == "FY"].copy()
    if fy.empty or m.empty:
        return pd.DataFrame()
    fy["eps"] = fy["items"].map(lambda d: d.get("net_income")) * fx / shares
    fy["bvps"] = fy["items"].map(lambda d: d.get("total_equity")) * fx / shares
    fy = fy.sort_values("usable_from")[["usable_from", "eps", "bvps"]]
    out = pd.merge_asof(m.sort_values("date"), fy, left_on="date", right_on="usable_from", direction="backward")
    out["per"] = np.where(out["eps"] > 0, out["close"] / out["eps"], np.nan)
    out["pbv"] = np.where(out["bvps"] > 0, out["close"] / out["bvps"], np.nan)
    return out.dropna(subset=["usable_from"])


def historical_value(hist: pd.DataFrame, items: dict, stype: str, shares: float, fx: float, min_months: int = 24) -> dict:
    if hist is None or hist.empty:
        return _res("historical", reason="histori harga/laporan tidak cukup")
    cv = lambda v: None if v is None else v * fx  # noqa: E731
    metrics = ["pbv"] if stype in FINANCIAL_TYPES | {"PROPERTY"} else ["per", "pbv"]
    lows, bases, highs, used = [], [], [], {}
    for metric in metrics:
        s = hist[metric].dropna()
        s = s[(s > 0) & (s < (100 if metric == "per" else 30))]
        if len(s) < min_months:
            continue
        base_val = cv(items.get("net_income" if metric == "per" else "total_equity"))
        if base_val is None or base_val <= 0:
            continue
        ps = base_val / shares
        q = s.quantile([0.25, 0.5, 0.75]).tolist()
        lows.append(q[0] * ps)
        bases.append(q[1] * ps)
        highs.append(q[2] * ps)
        used[metric] = {"p25": q[0], "median": q[1], "p75": q[2], "months": int(len(s))}
    if not bases:
        return _res("historical", reason=f"histori multiple valid < {min_months} bulan")
    return _res("historical", True, low=float(np.mean(lows)), base=float(np.mean(bases)), high=float(np.mean(highs)),
                detail=used)


# ------------------------------------------------------------------------------------------------------ DCF
def dcf_per_share(fcf0: float, growth: float, wacc: float, g_term: float, years: int, net_debt: float,
                  shares: float) -> float | None:
    if wacc <= g_term + 0.01 or shares <= 0:
        return None
    pv, f = 0.0, fcf0
    for t in range(1, years + 1):
        # pertumbuhan memudar linear dari `growth` ke g_term
        g = growth + (g_term - growth) * (t - 1) / max(years - 1, 1)
        f *= 1 + g
        pv += f / (1 + wacc) ** t
    tv = f * (1 + g_term) / (wacc - g_term)
    pv += tv / (1 + wacc) ** years
    return (pv - net_debt) / shares


def dcf_value(snap: dict, stype: str, shares: float, fx: float, market_cap: float | None, vcfg: dict,
              beta: float | None = None) -> dict:
    if stype in FINANCIAL_TYPES:
        return _res("dcf", reason="DCF FCFF tidak layak untuk bank/lembaga keuangan")
    if not shares or fx is None:
        return _res("dcf", reason="jumlah saham / kurs tidak tersedia")
    it, hist = snap.get("items", {}), snap.get("fy_history") or []
    fcfs = [h.get("free_cash_flow") if h.get("free_cash_flow") is not None else
            (None if h.get("operating_cash_flow") is None or h.get("capex") is None else h["operating_cash_flow"] - h["capex"])
            for h in hist]
    fcfs = [f for f in fcfs if f is not None]
    n_norm = 5 if stype == "COMMODITY" else 3
    if len(fcfs) < 3:
        return _res("dcf", reason="histori FCF < 3 tahun")
    fcf0 = float(np.mean(fcfs[-n_norm:])) * fx                    # FCF dinormalisasi (rata-rata), bukan 1 tahun
    if fcf0 <= 0:
        return _res("dcf", reason="FCF rata-rata ≤ 0 — DCF tidak bermakna")
    debt, cash = it.get("total_debt"), it.get("cash")
    if debt is None or cash is None:
        return _res("dcf", reason="utang/kas tidak tersedia")
    net_debt = (debt - cash) * fx
    g_term = float(vcfg.get("terminal_growth", 0.04))
    if stype == "COMMODITY":
        growth = g_term                                             # siklikal: tanpa ekstrapolasi pertumbuhan
    else:
        revs = [h.get("revenue") for h in hist if h.get("revenue")]
        growth = g_term
        if len(revs) >= 4 and revs[-4] > 0 and revs[-1] > 0:
            growth = (revs[-1] / revs[-4]) ** (1 / 3) - 1
        growth = float(np.clip(growth, 0.0, float(vcfg.get("max_explicit_growth", 0.15))))
    coe = cost_of_equity(vcfg, beta)
    kd = float(vcfg.get("cost_of_debt_pre_tax", 0.09)) * (1 - float(vcfg.get("tax_rate", 0.22)))
    d = max(debt * fx, 0.0)
    e = market_cap if market_cap and market_cap > 0 else None
    wacc = coe if e is None or d == 0 else (e * coe + d * kd) / (e + d)
    years = int(vcfg.get("explicit_years", 5))
    sens = vcfg.get("sensitivity", {}) or {}
    dr, dg = float(sens.get("discount_rate_step", 0.01)), float(sens.get("terminal_growth_step", 0.005))
    grid = {}
    for w in (wacc - dr, wacc, wacc + dr):
        for g in (g_term - dg, g_term, g_term + dg):
            grid[(round(w, 4), round(g, 4))] = dcf_per_share(fcf0, growth, w, g, years, net_debt, shares)
    base = grid.get((round(wacc, 4), round(g_term, 4)))
    low = dcf_per_share(fcf0, growth * 0.5, wacc + dr, g_term - dg, years, net_debt, shares)
    high = dcf_per_share(fcf0, min(growth * 1.25, float(vcfg.get("max_explicit_growth", 0.15))), wacc - dr, g_term + dg,
                         years, net_debt, shares)
    if base is None or base <= 0:
        return _res("dcf", reason="nilai ekuitas DCF ≤ 0 (utang bersih > nilai operasi)")
    return _res("dcf", True, low=max(low or 0.0, 0.0), base=base, high=max(high or base, base),
                assumptions={"fcf_normalized": fcf0, "explicit_growth": growth, "wacc": wacc, "cost_of_equity": coe,
                             "terminal_growth": g_term, "years": years, "net_debt": net_debt,
                             "normalization_years": min(n_norm, len(fcfs))},
                detail={"sensitivity": {f"wacc={w:.3f}|g={g:.3f}": v for (w, g), v in grid.items()}})


# ------------------------------------------------------------------------------------------------- dividend
def dividend_value(snap: dict, shares: float, fx: float, vcfg: dict, beta: float | None = None) -> dict:
    hist = snap.get("fy_history") or []
    if not shares or fx is None:
        return _res("dividend", reason="jumlah saham / kurs tidak tersedia")
    divs = [h.get("dividends_paid") for h in hist[-3:]]
    if len(divs) < 3 or any(d is None for d in divs):
        return _res("dividend", reason="data dividen 3 tahun tidak lengkap")
    if sum(1 for d in divs if d > 0) < 3:
        return _res("dividend", reason="dividen tidak konsisten 3 tahun")
    ni = [h.get("net_income") for h in hist[-3:]]
    if any(x is None or x <= 0 for x in ni) or sum(divs) / sum(ni) > 1.0:
        return _res("dividend", reason="payout > 100% atau laba ≤ 0 — dividen tidak berkelanjutan")
    coe = cost_of_equity(vcfg, beta)
    g = min(float(vcfg.get("terminal_growth", 0.04)), coe - 0.03)
    dps = float(np.mean(divs)) * fx / shares
    vals = [dps * (1 + gg) / (k - gg) for k, gg in ((coe + 0.01, g - 0.01), (coe, g), (coe - 0.01, g + 0.005))]
    return _res("dividend", True, low=vals[0], base=vals[1], high=vals[2],
                assumptions={"dps_avg_3y": dps, "cost_of_equity": coe, "growth": g, "payout_3y": sum(divs) / sum(ni)})


# --------------------------------------------------------------------------------------------- justified P/B
def justified_pb_value(snap: dict, stype: str, shares: float, fx: float, vcfg: dict, beta: float | None = None) -> dict:
    if stype not in FINANCIAL_TYPES:
        return _res("justified_pb", reason="hanya untuk bank/lembaga keuangan")
    hist = snap.get("fy_history") or []
    roes = [h["net_income"] / h["total_equity"] for h in hist[-3:]
            if h.get("net_income") is not None and (h.get("total_equity") or 0) > 0]
    eq = (snap.get("items") or {}).get("total_equity")
    if len(roes) < 2 or eq is None or eq <= 0 or not shares or fx is None:
        return _res("justified_pb", reason="ROE ≥ 2 tahun / ekuitas / jumlah saham tidak tersedia")
    roe = float(np.mean(roes))
    coe = cost_of_equity(vcfg, beta)
    g = min(float(vcfg.get("terminal_growth", 0.04)), coe - 0.03)
    if roe <= g:
        return _res("justified_pb", reason=f"ROE rata-rata {roe:.1%} ≤ pertumbuhan {g:.1%}")
    bvps = eq * fx / shares
    pb = lambda r, k: (r - g) / (k - g)  # noqa: E731
    vals = [pb(roe - 0.02, coe + 0.01) * bvps, pb(roe, coe) * bvps, pb(roe + 0.01, coe - 0.01) * bvps]
    return _res("justified_pb", True, low=max(vals[0], 0.0), base=vals[1], high=vals[2],
                assumptions={"roe_avg": roe, "cost_of_equity": coe, "growth": g, "bvps": bvps, "justified_pb": pb(roe, coe)})


# --------------------------------------------------------------------------------------------- kombinasi
def combine(methods: list[dict], weights: dict, min_weight: float = 0.4) -> dict:
    """Rata-rata berbobot metode valid (bobot dinormalisasi ulang). Bobot valid < min_weight → INSUFFICIENT_DATA."""
    valid = [m for m in methods if m["valid"] and weights.get(m["method"], 0) > 0 and m["base"] and m["base"] > 0]
    total = sum(weights.get(m["method"], 0) for m in valid)
    planned = sum(weights.values()) or 1.0
    coverage = total / planned
    if not valid or coverage < min_weight:
        return {"status": "INSUFFICIENT_DATA", "coverage": coverage, "n_methods": len(valid),
                "reason": "metode valid tidak cukup (" + ", ".join(f"{m['method']}: {m['reason']}" for m in methods
                                                              if not m["valid"]) + ")"}
    w = {m["method"]: weights[m["method"]] / total for m in valid}
    low = sum(w[m["method"]] * m["low"] for m in valid)
    base = sum(w[m["method"]] * m["base"] for m in valid)
    high = sum(w[m["method"]] * m["high"] for m in valid)
    bases = [m["base"] for m in valid]
    dispersion = max(bases) / min(bases) if min(bases) > 0 else np.inf
    if len(valid) >= 3 and coverage >= 0.7 and dispersion <= 1.6:
        conf = "HIGH"
    elif len(valid) >= 2 and dispersion <= 2.5:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    return {"status": "OK", "low": float(min(low, base)), "base": float(base), "high": float(max(high, base)),
            "coverage": coverage, "n_methods": len(valid), "dispersion": float(dispersion), "confidence": conf,
            "weights": w}
