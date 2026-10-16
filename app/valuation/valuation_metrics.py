"""Rasio fundamental & valuasi dengan aturan validitas eksplisit (§B2).

Setiap metrik mengembalikan nilai atau None + alasan di `invalid`. Contoh aturan:
- PER hanya bila laba bersih TTM > 0; PBV hanya bila ekuitas > 0;
- EV/EBITDA, EV/EBIT, EV/Sales, ROIC, net debt/EBITDA TIDAK dihitung untuk bank & lembaga keuangan
  (utang adalah bahan baku bisnisnya, EV tidak bermakna);
- P/FCF hanya bila FCF > 0; dividend yield hanya bila data dividen tersedia;
- semua metrik berbasis harga butuh jumlah saham & mata uang laporan = mata uang harga (atau kurs tersedia).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FINANCIAL_TYPES = {"BANK", "FINANCIAL"}

_BANK = ("bank",)
_FIN = ("financial", "keuangan", "insurance", "asuransi", "capital markets", "credit services", "multifinance",
        "pembiayaan", "investment", "asset management")
_COMMODITY = ("coal", "batubara", "batu bara", "oil", "gas", "minyak", "energy", "energi", "metal", "logam", "mining",
              "tambang", "gold", "emas", "nickel", "nikel", "copper", "tembaga", "plantation", "perkebunan", "palm",
              "sawit", "basic materials", "barang baku", "steel", "baja", "aluminum", "chemicals")


def sector_type(sector, subsector) -> str:
    """BANK | FINANCIAL | COMMODITY | PROPERTY | GENERAL | UNKNOWN — dari nama sektor GICS atau taksonomi Yahoo."""
    s = f"{sector or ''} {subsector or ''}".lower()
    if not s.strip() or s.strip() in ("unknown", "none unknown", "unknown unknown"):
        return "UNKNOWN"
    if any(k in s for k in _BANK):
        return "BANK"
    if any(k in s for k in _FIN):
        return "FINANCIAL"
    if any(k in s for k in _COMMODITY):
        return "COMMODITY"
    if any(k in s for k in ("real estate", "properti", "property")):
        return "PROPERTY"
    return "GENERAL"


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _sum_flow(rows: list[dict], key: str):
    vals = [r.get(key) for r in rows]
    return None if any(v is None for v in vals) else float(sum(vals))


def build_snapshot(statements: pd.DataFrame, tolerance: float = 0.05) -> dict | None:
    """Ringkasan fundamental 1 emiten dari laporan yang SUDAH diketahui (input sudah difilter point-in-time).

    Flow items (laba, arus kas): TTM dari 4 kuartal terakhir HANYA bila 4 kuartal yang berakhir di akhir FY terakhir
    terbukti cocok dengan angka FY (selisih revenue & laba ≤ tolerance) — membuktikan kuartal bersifat diskret.
    Jika tidak, flow items = FY terakhir. Stock items (neraca) = laporan terbaru.
    """
    if statements is None or statements.empty:
        return None
    s = statements.sort_values("period_end")
    fy = s[s["period_type"] == "FY"]
    q = s[s["period_type"].str.startswith("Q")]
    latest = s.iloc[-1]
    snap = {"currency": latest.get("currency"), "period_end": latest["period_end"],
            "known_date": s["usable_from"].max() if "usable_from" in s else s["first_known_date"].max(),
            "quality": sorted(set(s["quality_status"].dropna())), "basis": None}
    flow = None
    if len(fy):
        last_fy = fy.iloc[-1]
        flow, snap["basis"], snap["flow_period_end"] = dict(last_fy["items"]), "FY", last_fy["period_end"]
        q_after = q[q["period_end"] > last_fy["period_end"]]
        q_to_fy = q[(q["period_end"] <= last_fy["period_end"]) &
                    (q["period_end"] > last_fy["period_end"] - pd.DateOffset(months=12))]
        discrete = False
        if len(q_to_fy) == 4:
            for k in ("revenue", "net_income"):
                tot, fyv = _sum_flow(list(q_to_fy["items"]), k), last_fy["items"].get(k)
                if tot is None or fyv in (None, 0) or abs(tot - fyv) / abs(fyv) > tolerance:
                    break
            else:
                discrete = True
        snap["quarters_discrete"] = discrete
        if discrete and len(q_after):
            last4 = q.sort_values("period_end").tail(4)
            if (last4["period_end"].iloc[-1] - last4["period_end"].iloc[0]).days < 300:
                flow = {k: _sum_flow(list(last4["items"]), k) for k in last_fy["items"]}
                flow["eps_diluted"] = _sum_flow(list(last4["items"]), "eps_diluted")
                snap["basis"], snap["flow_period_end"] = "TTM", last4["period_end"].iloc[-1]
    elif len(q):
        snap["basis"] = "INSUFFICIENT"           # tanpa FY tidak bisa membuktikan sifat kuartal → tidak menebak TTM
    if flow is None:
        return snap | {"items": dict(latest["items"]), "flow_ok": False}
    stock_items = dict(latest["items"])
    items = {**flow, **{k: stock_items.get(k) for k in ("total_assets", "total_liabilities", "total_equity", "total_debt",
                                                         "cash", "current_assets", "current_liabilities", "shares_outstanding")}}
    snap.update(items=items, flow_ok=True)
    # histori tahunan untuk pertumbuhan & deteksi value trap
    hist = fy.tail(6)
    snap["fy_history"] = [{"period_end": r["period_end"], **r["items"]} for _, r in hist.iterrows()]
    return snap


def fundamental_ratios(snap: dict, stype: str) -> tuple[dict, dict]:
    """Rasio non-harga. Return (metrics, invalid_reasons)."""
    it = snap.get("items", {}) if snap else {}
    m, bad = {}, {}

    def put(name, val, reason=None):
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            bad[name] = reason or "komponen tidak tersedia"
        else:
            m[name] = float(val)

    rev, ni, eq = it.get("revenue"), it.get("net_income"), it.get("total_equity")
    put("gross_margin", _div(it.get("gross_profit"), rev) if rev and rev > 0 and stype not in FINANCIAL_TYPES else None,
        "tidak relevan untuk lembaga keuangan" if stype in FINANCIAL_TYPES else None)
    put("operating_margin", _div(it.get("operating_income"), rev) if rev and rev > 0 else None)
    put("net_margin", _div(ni, rev) if rev and rev > 0 else None)
    put("roe", _div(ni, eq) if eq and eq > 0 else None, "ekuitas ≤ 0" if eq is not None and eq <= 0 else None)
    put("roa", _div(ni, it.get("total_assets")) if (it.get("total_assets") or 0) > 0 else None)
    ocf, capex = it.get("operating_cash_flow"), it.get("capex")
    put("earnings_quality", _div(ocf, ni) if ni and ni > 0 and ocf is not None else None, "laba ≤ 0" if ni is not None and ni <= 0 else None)
    if stype in FINANCIAL_TYPES:
        for k in ("debt_to_equity", "net_debt_to_ebitda", "interest_coverage", "current_ratio", "roic"):
            bad[k] = "tidak relevan untuk bank/lembaga keuangan"
    else:
        debt, cash, ebitda, ebit = it.get("total_debt"), it.get("cash"), it.get("ebitda"), it.get("ebit")
        put("debt_to_equity", _div(debt, eq) if eq and eq > 0 else None, "ekuitas ≤ 0" if eq is not None and eq <= 0 else None)
        net_debt = None if debt is None or cash is None else debt - cash
        put("net_debt_to_ebitda", _div(net_debt, ebitda) if ebitda and ebitda > 0 else None, "EBITDA ≤ 0" if ebitda is not None and ebitda <= 0 else None)
        ie = it.get("interest_expense")
        put("interest_coverage", _div(ebit, ie) if ie and ie > 0 and ebit is not None else None)
        put("current_ratio", _div(it.get("current_assets"), it.get("current_liabilities")) if (it.get("current_liabilities") or 0) > 0 else None)
        tax_rate = 0.22
        if it.get("tax_expense") is not None and (it.get("pretax_income") or 0) > 0:
            tax_rate = min(max(it["tax_expense"] / it["pretax_income"], 0.0), 0.4)
        invested = None if None in (debt, eq, cash) else debt + eq - cash
        put("roic", _div(ebit * (1 - tax_rate), invested) if ebit is not None and invested and invested > 0 else None)
    put("fcf", (ocf - capex) if None not in (ocf, capex) else it.get("free_cash_flow"))
    # pertumbuhan dari histori FY (YoY & CAGR)
    h = snap.get("fy_history") or []
    if len(h) >= 2:
        a, b = h[-1], h[-2]
        for k, name in (("revenue", "revenue_growth"), ("net_income", "net_income_growth"), ("eps_diluted", "eps_growth"),
                        ("operating_cash_flow", "ocf_growth")):
            x0, x1 = b.get(k), a.get(k)
            put(name, (x1 / x0 - 1) if x0 and x1 is not None and x0 > 0 else None, "basis ≤ 0 atau tidak tersedia")
        om0 = _div(b.get("operating_income"), b.get("revenue")) if (b.get("revenue") or 0) > 0 else None
        om1 = _div(a.get("operating_income"), a.get("revenue")) if (a.get("revenue") or 0) > 0 else None
        put("margin_change", (om1 - om0) if None not in (om0, om1) else None)
        d0, d1 = _div(b.get("total_debt"), b.get("total_equity")), _div(a.get("total_debt"), a.get("total_equity"))
        if stype not in FINANCIAL_TYPES:
            put("leverage_change", (d1 - d0) if None not in (d0, d1) else None)
    if len(h) >= 4:
        r0, r1 = h[-4].get("revenue"), h[-1].get("revenue")
        put("revenue_cagr_3y", (r1 / r0) ** (1 / 3) - 1 if r0 and r1 and r0 > 0 and r1 > 0 else None)
    return m, bad


def price_multiples(snap: dict, stype: str, price: float, shares: float | None, fx: float | None = 1.0) -> tuple[dict, dict]:
    """Multiples berbasis harga. `fx` = harga satu unit mata uang laporan dalam mata uang harga (USD)."""
    it = snap.get("items", {}) if snap else {}
    m, bad = {}, {}
    if not price or price <= 0:
        return m, {"*": "harga tidak tersedia"}
    if not shares or shares <= 0:
        return m, {"*": "jumlah saham tidak tersedia"}
    if fx is None:
        return m, {"*": f"mata uang laporan {snap.get('currency')} ≠ USD dan kurs tidak tersedia"}
    cv = lambda v: None if v is None else v * fx  # noqa: E731 — konversi ke mata uang harga
    mcap = price * shares
    m["market_cap"] = mcap
    ni, eq, rev = cv(it.get("net_income")), cv(it.get("total_equity")), cv(it.get("revenue"))

    def put(name, num, den, cond, reason):
        if cond and num is not None and den:
            m[name] = num / den
        else:
            bad[name] = reason

    put("per", mcap, ni, ni is not None and ni > 0, "laba ≤ 0 atau tidak tersedia — PER tidak bermakna")
    put("pbv", mcap, eq, eq is not None and eq > 0, "ekuitas ≤ 0 atau tidak tersedia")
    fcf = cv(it.get("free_cash_flow"))
    put("p_fcf", mcap, fcf, fcf is not None and fcf > 0, "FCF ≤ 0 atau tidak tersedia")
    div = cv(it.get("dividends_paid"))
    if div is not None:
        m["dividend_yield"] = div / mcap
    else:
        bad["dividend_yield"] = "data dividen tidak tersedia"
    if stype in FINANCIAL_TYPES:
        for k in ("ev", "ev_ebitda", "ev_ebit", "ev_sales"):
            bad[k] = "EV tidak bermakna untuk bank/lembaga keuangan"
    else:
        debt, cash = cv(it.get("total_debt")), cv(it.get("cash"))
        if debt is None or cash is None:
            for k in ("ev", "ev_ebitda", "ev_ebit", "ev_sales"):
                bad[k] = "utang/kas tidak tersedia"
        else:
            ev = mcap + debt - cash
            m["ev"] = ev
            ebitda, ebit = cv(it.get("ebitda")), cv(it.get("ebit"))
            put("ev_ebitda", ev, ebitda, ebitda is not None and ebitda > 0 and ev > 0, "EBITDA ≤ 0 / EV ≤ 0")
            put("ev_ebit", ev, ebit, ebit is not None and ebit > 0 and ev > 0, "EBIT ≤ 0 / EV ≤ 0")
            put("ev_sales", ev, rev, rev is not None and rev > 0 and ev > 0, "pendapatan ≤ 0 / EV ≤ 0")
    if "per" in m:
        g = None
        h = snap.get("fy_history") or []
        if len(h) >= 4 and (h[-4].get("eps_diluted") or 0) > 0 and (h[-1].get("eps_diluted") or 0) > 0:
            g = (h[-1]["eps_diluted"] / h[-4]["eps_diluted"]) ** (1 / 3) - 1
        if g is not None and g > 0.02:                     # PEG hanya untuk pertumbuhan positif yang berarti
            m["peg"] = m["per"] / (100 * g)
        else:
            bad["peg"] = "pertumbuhan EPS 3 tahun ≤ 2% / tidak tersedia — PEG tidak layak"
    if fcf is not None and fcf > 0:
        m["fcf_yield"] = fcf / mcap
    return m, bad
