"""STOCK RESEARCH REPORT per emiten + CSV ranking riset.

`python main.py research TICKER [--date YYYY-MM-DD]` — dihitung ulang dari database, point-in-time pada tanggal tsb.
"""
from __future__ import annotations

import io
import json
import math

import pandas as pd

from app.database.repository import Repository
from app.research.integrated_scoring import RANKINGS, SCORE_COLS, assemble, rank_all

DISCLAIMER = ("Bukan nasihat investasi. Nilai wajar adalah estimasi berbasis asumsi (lihat config/research.yaml) "
              "dengan ketidakpastian besar; sinyal akumulasi adalah bukti perilaku harga-volume, bukan identitas pembeli. "
              "Belum ada bukti historis bahwa kombinasi skor ini menghasilkan return lebih baik (lihat evaluate-modules).")

CSV_COLS = ["ticker", "name", "sector", "close", "swing_decision", "value_decision", "valuation_status",
            "valuation_confidence", "fair_value_low", "fair_value_base", "fair_value_high", "margin_of_safety",
            *SCORE_COLS, "foreign_flow_status", "accumulation_status", "accumulation_stage"] + \
           [f"score_{r}" for r in RANKINGS] + [f"rank_{r}" for r in RANKINGS]


def research_csv(table: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    table.reindex(columns=[c for c in CSV_COLS if c in table]).sort_values(
        "rank_integrated" if "rank_integrated" in table else "ticker").to_csv(buf, index=False, float_format="%.4g")
    return buf.getvalue().encode("utf-8")


def _signals_from_db(repo: Repository, as_of) -> pd.DataFrame:
    """Rekonstruksi skor swing dari predictions + features snapshot (tanggal scan terakhir ≤ as_of)."""
    d = repo.db.scalar("SELECT MAX(prediction_date) FROM predictions WHERE prediction_date <= ?",
                       (pd.Timestamp(as_of).strftime("%Y-%m-%d"),))
    if not d:
        return pd.DataFrame(columns=["ticker"])
    d = str(d)[:10]
    pred = repo.db.query_df("SELECT p.ticker, p.decision, p.setup AS setup_type, p.market_regime, p.confidence, "
                            "p.close_price AS close, p.prob_bullish, p.score, p.stop_loss, p.tp1, p.tp2, p.risk_reward, "
                            "p.entry_low, p.entry_high, p.model_version, s.name, s.sector FROM predictions p "
                            "JOIN stocks s ON s.id = p.stock_id WHERE p.prediction_date = ?", (d,))
    feats = repo.db.query_df("SELECT s.ticker, f.payload FROM features f JOIN stocks s ON s.id = f.stock_id WHERE f.date = ?", (d,))
    if len(feats):
        pl = pd.DataFrame([json.loads(p) if isinstance(p, str) else {} for p in feats["payload"]])
        pl["ticker"] = feats["ticker"].values
        pred = pred.merge(pl[[c for c in ("ticker", "technical_score", "ml_score", "liquidity_score") if c in pl]],
                          on="ticker", how="left")
    pred.attrs["scan_date"] = d
    return pred


def build_research(cfg: dict, repo: Repository, as_of=None, tickers: list[str] | None = None) -> dict:
    from app.accumulation.accumulation_report import run_accumulation, run_flow_analysis
    from app.valuation.valuation_report import run_valuation
    as_of = pd.Timestamp(as_of) if as_of is not None else repo.max_price_date()
    if as_of is None:
        raise ValueError("Tidak ada data harga di database")
    prices = repo.load_prices(start=(as_of - pd.Timedelta(days=400)).strftime("%Y-%m-%d"), end=as_of.strftime("%Y-%m-%d"))
    sig = _signals_from_db(repo, as_of)
    if sig.empty:
        last = prices.sort_values("date").groupby("ticker").tail(1)
        sig = last[["ticker", "close"]].assign(decision="NO_SCAN")
    val = run_valuation(cfg, repo, as_of, prices)
    p_sub = prices if not tickers else prices[prices["ticker"].isin(tickers)]
    flow = run_flow_analysis(cfg, repo, p_sub, as_of)
    flow_last = flow.sort_values("date").groupby("ticker").tail(1) if len(flow) else None
    acc = run_accumulation(cfg, repo, p_sub, as_of, flow)
    table = rank_all(assemble(sig, val, flow_last, acc), cfg)
    return {"as_of": as_of, "table": table, "valuation": val, "flow": flow_last, "accumulation": acc, "signals": sig}


def _v(x, fmt="{:,.2f}"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or (not isinstance(x, str) and pd.isna(x)):
        return "UNAVAILABLE"
    return fmt.format(x) if not isinstance(x, str) else x


def stock_report(res: dict, ticker: str) -> str:
    from app.accumulation.accumulation_report import format_accumulation
    from app.flows.flow_report import format_flow
    from app.valuation.valuation_report import format_valuation
    t = res["table"]
    if ticker not in set(t["ticker"]):
        raise ValueError(f"{ticker} tidak ada di universe aktif / tidak punya harga")
    row = t[t["ticker"] == ticker].iloc[0].to_dict()
    sig = res["signals"]
    srow = sig[sig["ticker"] == ticker].iloc[0].to_dict() if ticker in set(sig["ticker"]) else {}
    vrow = res["valuation"][res["valuation"]["ticker"] == ticker]
    frow = res["flow"][res["flow"]["ticker"] == ticker] if res["flow"] is not None else pd.DataFrame()
    arow = res["accumulation"][res["accumulation"]["ticker"] == ticker] if res["accumulation"] is not None else pd.DataFrame()
    n = len(t)
    L = ["=" * 78, f"STOCK RESEARCH REPORT — {ticker}  {row.get('name') or ''}".rstrip(),
         f"Tanggal data         : {res['as_of']:%Y-%m-%d}   Sektor: {row.get('sector') or 'UNKNOWN'}",
         f"Harga penutupan      : {_v(row.get('close'), '${:,.2f}')}", "=" * 78,
         "", "[KEPUTUSAN — VALUE dan SWING dinilai TERPISAH]",
         f"VALUE decision       : {row.get('value_decision')}",
         f"SWING decision       : {row.get('swing_decision') or srow.get('decision') or 'NO_SCAN'}"
         f"  (scan {sig.attrs.get('scan_date', '-')}, setup {srow.get('setup_type') or '-'})"]
    if srow.get("decision") in ("BUY", "WATCHLIST"):
        usd = "${:,.2f}"
        L.append(f"Swing plan           : entry {_v(srow.get('entry_low'), usd)}–{_v(srow.get('entry_high'), usd)}, SL "
                 f"{_v(srow.get('stop_loss'), usd)}, TP {_v(srow.get('tp1'), usd)}/{_v(srow.get('tp2'), usd)}, "
                 f"RR {_v(srow.get('risk_reward'), '{:.2f}')}")
    L += ["", "[VALUATION]"] + (format_valuation(vrow.iloc[0].to_dict()) if len(vrow) else ["UNAVAILABLE"])
    L += ["", "[SHORT FLOW — FINRA]"] + format_flow(frow.iloc[0].to_dict() if len(frow) else None)
    L += ["", "[ACCUMULATION / DISTRIBUTION]"] + format_accumulation(arow.iloc[0].to_dict() if len(arow) else None)
    L += ["", "[SKOR TERPISAH 0–100]"]
    for c in SCORE_COLS:
        val = row.get(c)
        L.append(f"  {c:<26}: {_v(val, '{:.0f}') if c != 'value_trap_risk' else val}")
    L += ["", f"[RANKING dari {n} emiten — kosong = data tidak cukup untuk diranking]"]
    for r in RANKINGS:
        rk, sc = row.get(f"rank_{r}"), row.get(f"score_{r}")
        L.append(f"  {r:<20}: {'#' + str(int(rk)) if rk is not None and not pd.isna(rk) else '-':>6}  "
                 f"skor {_v(sc, '{:.1f}')}")
    L += ["", "[CATATAN]", "  " + DISCLAIMER, "=" * 78]
    return "\n".join(L)
