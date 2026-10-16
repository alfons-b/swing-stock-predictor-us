"""Orkestrasi valuasi seluruh universe pada satu tanggal + penyimpanan + teks laporan.

run_valuation(cfg, repo, as_of) — SEMUA input difilter point-in-time:
  - laporan keuangan: hanya versi yang first_known_date ≤ as_of (atau estimated_available_date bila mode riset);
  - harga: penutupan terakhir ≤ as_of;
  - kurs: kurs terakhir ≤ as_of.
Satu sumber laporan per emiten (prioritas provider) — angka antar sumber tidak dicampur.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from app.config import get, resolve_path
from app.database.repository import Repository, now_utc
from app.utils.logging_utils import get_logger
from app.valuation import intrinsic_value as iv
from app.valuation.fundamental_provider import load_statements
from app.valuation.margin_of_safety import classify, margin_of_safety
from app.valuation.sector_comparison import peer_group, peer_percentile, peer_quartiles
from app.valuation.valuation_metrics import build_snapshot, fundamental_ratios, price_multiples, sector_type
from app.valuation.valuation_scorer import quality_score, value_score
from app.valuation.value_trap_detector import detect

log = get_logger(__name__)
PEER_METRICS = ("per", "pbv", "ev_ebitda")


def _clean(o):
    """JSON aman: NaN/inf → None, Timestamp → string."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return None if pd.isna(o) else o.strftime("%Y-%m-%d")
    return o


def load_fx(cfg: dict) -> pd.DataFrame:
    """Kurs: CSV date,currency,rate — `rate` = harga 1 unit `currency` dalam mata uang harga (USD).
    Nama kolom `rate_usd` (atau `rate_idr` dari versi BEI) juga diterima."""
    p = resolve_path(cfg, str(get(cfg, "fundamentals.fx_file", "data/raw/fx/fx_rates.csv")))
    empty = pd.DataFrame(columns=["date", "currency", "rate"])
    if not p.exists():
        return empty
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    for alt in ("rate_usd", "rate_idr"):
        if "rate" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "rate"})
    if not {"date", "currency", "rate"} <= set(df.columns):
        log.warning("File kurs %s butuh kolom date,currency,rate — diabaikan", p)
        return empty
    df["date"] = pd.to_datetime(df["date"])
    df["currency"] = df["currency"].str.upper().str.strip()
    return df.dropna(subset=["rate"])


def fx_rate(fx: pd.DataFrame, currency: str | None, as_of, price_ccy: str = "USD") -> float | None:
    """Kurs 1 unit `currency` dalam mata uang harga yang diketahui pada as_of. Tidak diketahui → None (tidak ditebak)."""
    if not currency:
        return None
    if currency.upper() == price_ccy.upper():
        return 1.0
    r = fx[(fx["currency"] == currency.upper()) & (fx["date"] <= pd.Timestamp(as_of))]
    if r.empty or (pd.Timestamp(as_of) - r["date"].max()).days > 10:
        return None
    return float(r.sort_values("date")["rate"].iloc[-1])


def _source_rank(cfg) -> dict:
    return {e.get("name"): int(e.get("priority", 50)) for e in get(cfg, "fundamentals.providers", []) or []}


def pick_source(st: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Satu sumber per emiten: prioritas provider terkecil yang punya laporan FY; jika tidak ada, prioritas terkecil."""
    if st.empty:
        return st
    rank = _source_rank(cfg)
    st = st.assign(_rank=st["source"].map(lambda s: rank.get(s, 50)), _fy=(st["period_type"] == "FY"))
    best = (st.groupby(["stock_id", "source"]).agg(rank=("_rank", "first"), has_fy=("_fy", "any")).reset_index()
            .sort_values(["stock_id", "has_fy", "rank"], ascending=[True, False, True]).drop_duplicates("stock_id"))
    keep = set(zip(best["stock_id"], best["source"]))
    return st[[k in keep for k in zip(st["stock_id"], st["source"])]].drop(columns=["_rank", "_fy"])


def monthly_closes(repo: Repository, as_of, years: int) -> pd.DataFrame:
    """Penutupan akhir bulan (hari bursa terakhir tiap bulan) — ringan dibanding memuat seluruh histori harian."""
    start = (pd.Timestamp(as_of) - pd.DateOffset(years=years, months=1)).strftime("%Y-%m-%d")
    end = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    dates = repo.db.query_df("SELECT DISTINCT date FROM price_history WHERE date >= ? AND date <= ?", (start, end),
                             parse_dates=["date"])
    if dates.empty:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    d = dates["date"].sort_values()
    month_ends = d.groupby(d.dt.to_period("M")).max().dt.strftime("%Y-%m-%d").tolist()
    ph = ",".join(["?"] * len(month_ends))
    df = repo.db.query_df(f"SELECT s.ticker, p.date, p.close FROM price_history p JOIN stocks s ON s.id = p.stock_id "
                          f"WHERE p.date IN ({ph})", month_ends, parse_dates=["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def run_valuation(cfg: dict, repo: Repository, as_of, prices: pd.DataFrame | None = None,
                  use_estimated: bool = False, run_id: str | None = None) -> pd.DataFrame:
    """prices: harga harian (ticker,date,close) ≥ ±250 hari terakhir (untuk deteksi falling knife). None → dimuat."""
    vcfg = cfg.get("valuation", {}) or {}
    as_of = pd.Timestamp(as_of).normalize()
    stocks = repo.stocks(active_only=True)
    if stocks.empty:
        return pd.DataFrame()
    if prices is None:
        prices = repo.load_prices(start=(as_of - pd.Timedelta(days=400)).strftime("%Y-%m-%d"), end=as_of.strftime("%Y-%m-%d"))
    prices = prices[prices["date"] <= as_of]
    last_px = prices.sort_values("date").groupby("ticker")["close"].last()
    st_all = load_statements(repo, as_of, use_estimated=use_estimated)
    st_all = pick_source(st_all, cfg) if len(st_all) else st_all
    by_stock = {sid: g for sid, g in st_all.groupby("stock_id")} if len(st_all) else {}
    fx = load_fx(cfg)
    price_ccy = get(cfg, "fundamentals.price_currency", "USD")
    monthly = monthly_closes(repo, as_of, int(vcfg.get("history_years", 5)))
    monthly_by = {t: g for t, g in monthly.groupby("ticker")} if len(monthly) else {}
    px_by = {t: g for t, g in prices.groupby("ticker")}
    stale_days = int(get(cfg, "fundamentals.stale_after_days", 450))

    # ---- tahap 1: snapshot + rasio per emiten (untuk kuartil peer)
    base = []
    for s in stocks.itertuples():
        stype = sector_type(s.sector, s.subsector)
        stmts = by_stock.get(s.id)
        snap = build_snapshot(stmts) if stmts is not None else None
        price = last_px.get(s.ticker)
        rec = {"stock_id": s.id, "ticker": s.ticker, "sector": s.sector, "subsector": s.subsector, "sector_type": stype,
               "price": None if price is None or pd.isna(price) else float(price), "snap": snap, "stmts": stmts}
        if snap and snap.get("flow_ok"):
            shares = (snap["items"].get("shares_outstanding") or
                      (float(s.listed_shares) if s.listed_shares is not None and not pd.isna(s.listed_shares) else None))
            rate = fx_rate(fx, snap.get("currency"), as_of, price_ccy)
            ratios, bad_r = fundamental_ratios(snap, stype)
            mult, bad_m = price_multiples(snap, stype, rec["price"], shares, rate)
            rec.update(shares=shares, fx=rate, ratios=ratios, multiples=mult, invalid={**bad_r, **bad_m})
            rec.update({k: mult.get(k) for k in PEER_METRICS})
        base.append(rec)
    uni = pd.DataFrame([{k: r.get(k) for k in ("ticker", "sector", "subsector", "sector_type", *PEER_METRICS)} for r in base])
    min_peers = int(vcfg.get("min_peers", 5))
    quart = {m: peer_quartiles(uni, m, min_peers) for m in PEER_METRICS}

    # ---- tahap 2: metode nilai wajar + MoS + value trap + skor
    out = []
    weights_all = vcfg.get("method_weights", {}) or {}
    for r in base:
        snap, stype, price = r["snap"], r["sector_type"], r["price"]
        row = {"stock_id": r["stock_id"], "ticker": r["ticker"], "as_of_date": as_of.strftime("%Y-%m-%d"), "price": price,
               "sector_type": stype, "run_id": run_id, "created_at": now_utc()}
        if not snap or not snap.get("flow_ok"):
            reason = "tidak ada laporan keuangan yang diketahui pada tanggal ini" if not snap else \
                f"basis laporan {snap.get('basis')} — tidak ada laporan tahunan (FY)"
            out.append(row | {"valuation_status": "INSUFFICIENT_DATA", "value_trap_risk": "UNKNOWN",
                              "data_status": "NO_FUNDAMENTALS" if not snap else "INSUFFICIENT_HISTORY",
                              "methods": json.dumps({"reason": reason})})
            continue
        shares, rate = r.get("shares"), r.get("fx")
        peers = {}
        for m in PEER_METRICS:
            pg = peer_group(r, quart[m])
            if pg:
                peers[m] = pg[2] | {"level": pg[0], "group": pg[1]}
        mcap = (r.get("multiples") or {}).get("market_cap")
        methods = []
        if price is None or shares is None or rate is None:
            why = ("harga tidak tersedia" if price is None else "jumlah saham tidak tersedia" if shares is None
                   else f"mata uang laporan {snap.get('currency')} tanpa kurs ke {price_ccy}")
            methods = [iv._res(m, reason=why) for m in ("relative", "historical", "dcf", "dividend", "justified_pb")]
        else:
            hist = iv.historical_multiples(monthly_by.get(r["ticker"]), r["stmts"], shares, rate,
                                           int(vcfg.get("history_years", 5)))
            methods = [iv.relative_value(snap["items"], stype, shares, rate, peers),
                       iv.historical_value(hist, snap["items"], stype, shares, rate, int(vcfg.get("min_history_months", 24))),
                       iv.dcf_value(snap, stype, shares, rate, mcap, vcfg),
                       iv.dividend_value(snap, shares, rate, vcfg),
                       iv.justified_pb_value(snap, stype, shares, rate, vcfg)]
        weights = weights_all.get(stype) or weights_all.get("UNKNOWN") or {"relative": 0.5, "historical": 0.5}
        comb = iv.combine(methods, weights, float(vcfg.get("min_method_weight", 0.4)))
        trap = detect(snap, r.get("ratios") or {}, stype, px_by.get(r["ticker"]), as_of, stale_days)
        q_score, q_comp = quality_score(r.get("ratios") or {}, stype)
        metrics = {**(r.get("ratios") or {}), **(r.get("multiples") or {})}
        row.update(metrics=json.dumps(_clean({"values": metrics, "invalid": r.get("invalid"), "quality_components": q_comp,
                                              "basis": snap.get("basis"), "currency": snap.get("currency"), "fx": rate,
                                              "shares": shares})),
                   methods=json.dumps(_clean({"methods": methods, "combined": comb, "weights_planned": weights})),
                   value_trap_risk=trap["risk"], value_trap_reasons=json.dumps(_clean(trap)), quality_score=q_score,
                   fundamentals_known_date=_clean(pd.Timestamp(snap["known_date"])) if snap.get("known_date") is not None else None,
                   fundamentals_period_end=_clean(pd.Timestamp(snap.get("flow_period_end") or snap["period_end"])))
        if comb["status"] != "OK" or price is None:
            row.update(valuation_status="INSUFFICIENT_DATA", data_status="INSUFFICIENT_METHODS", value_score=None)
        else:
            mos, mos_c = margin_of_safety(comb["base"], price), margin_of_safety(comb["low"], price)
            pp = None
            if "per" in peers and r.get("per") is not None:
                pp = peer_percentile(uni, r["ticker"], "per", peers["per"]["level"])
            elif "pbv" in peers and r.get("pbv") is not None:
                pp = peer_percentile(uni, r["ticker"], "pbv", peers["pbv"]["level"])
            row.update(fair_value_low=comb["low"], fair_value_base=comb["base"], fair_value_high=comb["high"],
                       margin_of_safety=mos, margin_of_safety_conservative=mos_c, valuation_confidence=comb["confidence"],
                       valuation_status=classify(mos, mos_c, comb["confidence"], vcfg.get("status_thresholds", {})),
                       value_score=value_score(mos, mos_c, comb["confidence"], pp),
                       data_status="ESTIMATED_PIT" if use_estimated else "OK")
        out.append(row)
    df = pd.DataFrame(out)
    for c in VALUATION_COLS:                                   # kolom selalu ada (nilai None bila tidak dihitung)
        if c not in df:
            df[c] = None
    if len(df):
        n_ok = int((df["valuation_status"] != "INSUFFICIENT_DATA").sum())
        log.info("Valuasi %s: %d/%d emiten punya nilai wajar", as_of.date(), n_ok, len(df), extra={"persist": True})
    return df


VALUATION_COLS = ["stock_id", "as_of_date", "price", "sector_type", "fair_value_low", "fair_value_base", "fair_value_high",
                  "margin_of_safety", "margin_of_safety_conservative", "valuation_status", "valuation_confidence",
                  "value_score", "quality_score", "value_trap_risk", "methods", "metrics", "value_trap_reasons",
                  "fundamentals_known_date", "fundamentals_period_end", "data_status", "run_id", "created_at"]


def compact_json(row_methods: str | None, row_metrics: str | None, row_trap: str | None) -> tuple:
    """Versi ringkas untuk database (kuota storage paket gratis): tanpa grid sensitivitas & teks alasan per metrik.
    Detail lengkap selalu bisa dihitung ulang point-in-time (`python main.py research TICKER --date ...`)."""
    m = json.loads(row_methods) if isinstance(row_methods, str) else {}
    for x in m.get("methods", []):
        if x.get("method") == "dcf":
            x.pop("detail", None)
        if x.get("method") in ("relative", "historical"):
            x["detail"] = {k: {kk: vv for kk, vv in (v or {}).items() if kk in ("peer_median", "median", "n_peers", "months",
                                                                                "peer_group")}
                           for k, v in (x.get("detail") or {}).items()}
        if x.get("valid"):
            x.pop("reason", None)
    met = json.loads(row_metrics) if isinstance(row_metrics, str) else {}
    if isinstance(met.get("invalid"), dict):
        met["invalid"] = sorted(met["invalid"])
    trap = json.loads(row_trap) if isinstance(row_trap, str) else {}
    trap.pop("checks", None)
    return (json.dumps(m, separators=(",", ":")) if m else None, json.dumps(met, separators=(",", ":")) if met else None,
            json.dumps(trap, separators=(",", ":")) if trap else None)


def save_valuations(repo: Repository, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    d = df.reindex(columns=VALUATION_COLS).copy()
    packed = [compact_json(a, b, c) for a, b, c in zip(d["methods"], d["metrics"], d["value_trap_reasons"])]
    d["methods"], d["metrics"], d["value_trap_reasons"] = zip(*packed) if packed else ([], [], [])
    repo.db.upsert("valuation_results", d, keys=["stock_id", "as_of_date"], count=False)
    return len(d)


def latest_valuations(repo: Repository, as_of=None) -> pd.DataFrame:
    """Valuasi terbaru per emiten (≤ as_of)."""
    sql = ("SELECT v.*, s.ticker, s.name, s.sector FROM valuation_results v JOIN stocks s ON s.id = v.stock_id "
           "WHERE v.as_of_date = (SELECT MAX(as_of_date) FROM valuation_results" + (" WHERE as_of_date <= ?" if as_of else "") + ")")
    return repo.db.query_df(sql, (pd.Timestamp(as_of).strftime("%Y-%m-%d"),) if as_of else (), parse_dates=["as_of_date"])


def _fmt(v, pct=False, digits=0):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "UNAVAILABLE"
    return f"{v:.1%}" if pct else f"{v:,.{digits}f}"


def _usd(v) -> str:
    out = _fmt(v, digits=2)
    return out if out == "UNAVAILABLE" else f"${out}"


def format_valuation(row: dict) -> list[str]:
    """Baris teks bagian VALUATION untuk STOCK RESEARCH REPORT."""
    methods = json.loads(row.get("methods") or "{}") if isinstance(row.get("methods"), str) else (row.get("methods") or {})
    metrics = json.loads(row.get("metrics") or "{}") if isinstance(row.get("metrics"), str) else (row.get("metrics") or {})
    trap = json.loads(row.get("value_trap_reasons") or "{}") if isinstance(row.get("value_trap_reasons"), str) \
        else (row.get("value_trap_reasons") or {})
    v = metrics.get("values", {})
    lines = [f"Sector type          : {row.get('sector_type')}  (basis laporan: {metrics.get('basis', '-')}, "
             f"mata uang: {metrics.get('currency', '-')})",
             f"Fundamental known    : {row.get('fundamentals_known_date') or '-'} (periode {row.get('fundamentals_period_end') or '-'})",
             f"PER / PBV            : {_fmt(v.get('per'), digits=1)} / {_fmt(v.get('pbv'), digits=2)}",
             f"EV/EBITDA            : {_fmt(v.get('ev_ebitda'), digits=1)}",
             f"ROE / Net margin     : {_fmt(v.get('roe'), True)} / {_fmt(v.get('net_margin'), True)}",
             f"Dividend yield       : {_fmt(v.get('dividend_yield'), True)}",
             f"Fair value (L/B/H)   : {_usd(row.get('fair_value_low'))} / {_usd(row.get('fair_value_base'))} / "
             f"{_usd(row.get('fair_value_high'))}",
             f"Margin of safety     : {_fmt(row.get('margin_of_safety'), True)} (konservatif "
             f"{_fmt(row.get('margin_of_safety_conservative'), True)})",
             f"Valuation status     : {row.get('valuation_status')}  confidence {row.get('valuation_confidence') or '-'}",
             f"Value score          : {_fmt(row.get('value_score'), digits=0)}   Quality score: {_fmt(row.get('quality_score'), digits=0)}",
             f"Value trap risk      : {row.get('value_trap_risk')}"]
    for r in trap.get("reasons", [])[:6]:
        lines.append(f"  - {r}")
    for m in methods.get("methods", []):
        state = (f"{_usd(m.get('low'))} / {_usd(m.get('base'))} / {_usd(m.get('high'))}"
                 if m.get("valid")
                 else f"tidak valid: {m.get('reason')}")
        lines.append(f"  [{m.get('method'):<12}] {state}")
    if methods.get("reason"):
        lines.append(f"  {methods['reason']}")
    return lines
