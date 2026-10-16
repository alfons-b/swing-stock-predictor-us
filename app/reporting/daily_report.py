"""Daily report (§42): data/output/daily_predictions_YYYYMMDD.csv + reports/daily_report_YYYYMMDD.{html,md}."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from app.reporting.html_report import CSS, REGIME_LABEL, regime_gauge, stock_chart_svg

CSV_COLS = ["date", "ticker", "name", "sector", "decision", "setup_type", "final_score", "probability_bullish",
            "probability_neutral", "probability_bearish", "expected_return_5d", "prob_hit_tp", "close", "entry_low",
            "entry_ideal", "entry_high", "stop_loss", "take_profit_1", "take_profit_2", "risk_reward", "confidence",
            "position_lots", "position_size", "capital_required", "estimated_loss", "market_regime", "sector_score"]


def esc(x):
    return html.escape("" if x is None else str(x))


def rp(x):
    """Format USD: $1,234.56 (harga & nilai uang di pasar AS memakai 2 desimal)."""
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"${x:,.2f}"


def pc(x, signed=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.0f}%"


def predictions_csv(sig: pd.DataFrame, horizon: int) -> bytes:
    d = sig.copy()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    cols = [c.replace("_5d", f"_{horizon}d") for c in CSV_COLS]
    d["reject_reasons"] = d["reject_reasons"].map(lambda r: ";".join(r))
    order = {"BUY": 0, "WATCHLIST": 1, "WAIT": 2, "AVOID": 3}
    d = d.assign(_o=d["decision"].map(order)).sort_values(["_o", "final_score"], ascending=[True, False])
    return d.reindex(columns=cols + ["reject_reasons"]).to_csv(index=False).encode("utf-8")


def sector_table(sig: pd.DataFrame) -> pd.DataFrame:
    s = sig.groupby("sector").agg(score=("sector_score", "first"), ret5=("sector_ret5", "first"), ret20=("sector_ret20", "first"),
                                  ret60=("sector_ret60", "first"), rs20=("sector_rs20", "first"), breadth=("sector_breadth", "first"),
                                  volume_mom=("sector_volume_mom", "first"), n_stocks=("ticker", "count"))
    s = s.sort_values("score", ascending=False)
    s["rank"] = np.arange(1, len(s) + 1)
    return s.reset_index()


def markdown_report(summary: dict, sectors: pd.DataFrame) -> str:
    L = [f"# Daily Swing Report — {summary['date']}", ""]
    if summary.get("data_warning"):
        L += [f"> ⚠️ {summary['data_warning']}", ""]
    f = summary["data_freshness"]
    L += [f"**Market regime:** {summary['market_regime']} · **Data:** {f['status']} (DB {f['latest_database_date']}, "
          f"ekspektasi {f['latest_expected_market_date']}) · **Model:** {summary['model_version']}", "",
          f"## {summary['headline']}", ""]
    if summary["recommendations"]:
        L += ["| # | Ticker | Setup | Skor | P(bull) | E[ret] | Entry | SL | TP1 | TP2 | R/R | Qty | Conf. | Decision |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in summary["recommendations"]:
            er = next(v for k, v in r.items() if k.startswith("expected_return_"))
            L.append(f"| {r['rank']} | {r['ticker']} | {r['setup']} | {r['score']:.0f} | {pc(r['probability_bullish'], False)} | "
                     f"{pc(er)} | {rp(r['entry_low'])}–{rp(r['entry_high'])} | {rp(r['stop_loss'])} | {rp(r['take_profit_1'])} | "
                     f"{rp(r['take_profit_2'])} | {r['risk_reward']:.2f} | {r['lots']} | {r['confidence']} | {r['decision']} |")
        L.append("")
        for r in summary["recommendations"]:
            L += [f"### {r['rank']}. {r['ticker']}", "Alasan: " + "; ".join(r["reasons"]), "",
                  "Risiko: " + ("; ".join(r["risks"]) or "-"), ""]
    if summary.get("watchlist"):
        L += ["## Watchlist", ""] + [f"- {w['ticker']} ({w['setup']}, skor {w['score']}) — tertahan: {', '.join(w['blocked_by'])}"
                                    for w in summary["watchlist"]] + [""]
    L += ["## Sector ranking", "", "| # | Sektor | Skor | 5H | 20H | 60H | RS 20H | Breadth |", "|---|---|---|---|---|---|---|---|"]
    for r in sectors.itertuples():
        L.append(f"| {r.rank} | {r.sector} | {r.score:.0f} | {pc(r.ret5)} | {pc(r.ret20)} | {pc(r.ret60)} | {pc(r.rs20)} | {pc(r.breadth, False)} |")
    if summary.get("coverage"):
        from app.scanner.coverage import coverage_markdown
        L += [""] + coverage_markdown(summary["coverage"])
    L += ["", f"_{summary['disclaimer']}_"]
    return "\n".join(L)


def html_report(summary: dict, sectors: pd.DataFrame, df: pd.DataFrame, evaluation: dict | None = None) -> str:
    P = [f"<h1>Swing scan US · {esc(pd.Timestamp(summary['date']).strftime('%d %B %Y'))}</h1>"]
    if summary.get("data_warning"):
        P.append(f'<p class="warn"><b>Data contoh sintetis.</b> {esc(summary["data_warning"])}</p>')
    f = summary["data_freshness"]
    if not f.get("trading_allowed", True):
        P.append(f'<p class="warn"><b>Data {esc(f["status"])}.</b> Database terakhir {esc(f["latest_database_date"])}, '
                 f'seharusnya {esc(f["latest_expected_market_date"])}. Rekomendasi BUY ditahan.</p>')
    P.append(regime_gauge(summary["market_regime"]))
    mc = summary["market_context"]
    P.append(f'<div class="facts"><span>Regime S&P 500 <b>{REGIME_LABEL.get(summary["market_regime"], summary["market_regime"])}</b></span>'
             f'<span>S&P 500 20 hari <b>{pc(mc.get("ihsg_ret20"))}</b></span><span>Di atas SMA50 <b>{pc(mc.get("breadth_above_sma50"), False)}</b></span>'
             f'<span>Data <b>{esc(f["status"])}</b> ({esc(f["latest_database_date"])})</span>'
             f'<span>Discan <b>{summary["universe_scanned"]}</b>, layak <b>{summary["tradeable"]}</b></span>'
             f'<span>Model <b>{esc(summary["model_version"])}</b></span></div>')
    P.append(f"<h2>{esc(summary['headline'].capitalize() if not summary['recommendations'] else summary['headline'].title())}</h2>")
    recs = summary["recommendations"]
    if not recs:
        P.append('<p class="empty">Tidak ada saham yang lolos semua filter risiko hari ini. Menunggu adalah posisi yang sah.</p>')
    else:
        rows = []
        for r in recs:
            er = next(v for k, v in r.items() if k.startswith("expected_return_"))
            rows.append(f'<tr><td>{r["rank"]}</td><td class="l"><b>{esc(r["ticker"])}</b></td><td class="l">{esc(r["setup"])}</td>'
                        f'<td>{r["score"]:.0f}</td><td>{pc(r["probability_bullish"], False)}</td><td>{pc(er)}</td>'
                        f'<td>{rp(r["entry_low"])}–{rp(r["entry_high"])}</td><td>{rp(r["stop_loss"])}</td><td>{rp(r["take_profit_1"])}</td>'
                        f'<td>{rp(r["take_profit_2"])}</td><td>{r["risk_reward"]:.2f}</td><td>{r["lots"]}</td>'
                        f'<td class="l">{r["confidence"]}</td><td class="l"><span class="dec {r["decision"]}">{r["decision"]}</span></td></tr>')
        P.append('<div class="scroll"><table><thead><tr><th>#</th><th class="l">Saham</th><th class="l">Setup</th><th>Skor</th><th>P(bull)</th>'
                 '<th>E[ret]</th><th>Entry</th><th>SL</th><th>TP1</th><th>TP2</th><th>R/R</th><th>Qty</th><th class="l">Conf.</th>'
                 f'<th class="l">Keputusan</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
        for r in recs[:5]:
            bars = df[(df["ticker"] == r["ticker"]) & (df["date"] <= pd.Timestamp(summary["date"]))].sort_values("date")
            last = bars.iloc[-1]
            lv = {"entry": r["entry_ideal"], "stop": r["stop_loss"], "tp1": r["take_profit_1"], "tp2": r["take_profit_2"],
                  "sup": float(last.get("sup20", np.nan)), "res": float(last.get("res50", np.nan))}
            pros = "".join(f'<li class="pro">{esc(x)}</li>' for x in r["reasons"])
            cons = "".join(f'<li class="con">{esc(x)}</li>' for x in r["risks"]) or '<li class="muted">Tidak ada yang menonjol</li>'
            P.append(f'<article class="pick"><header><div><h3>{r["rank"]}. {esc(r["ticker"])} — {esc(r.get("name"))}</h3>'
                     f'<span class="muted">{esc(r.get("sector"))} · {esc(r["setup"])} · confidence {r["confidence"]}</span></div>'
                     f'<div class="score">{r["score"]:.0f}<span class="muted" style="font-size:14px">/100</span></div></header>'
                     f'<div class="levels"><div>Entry<b>{rp(r["entry_low"])}–{rp(r["entry_high"])}</b></div><div>Stop loss<b>{rp(r["stop_loss"])}</b></div>'
                     f'<div>TP1 / TP2<b>{rp(r["take_profit_1"])} / {rp(r["take_profit_2"])}</b></div><div>R/R<b>{r["risk_reward"]:.2f}</b></div>'
                     f'<div>Posisi<b>{r["lots"]} saham</b></div><div>Modal<b>{rp(r["capital_required"])}</b></div></div>'
                     f'{stock_chart_svg(bars, lv)}<div class="cols"><div><h3>Alasan</h3><ul>{pros}</ul></div>'
                     f'<div><h3>Risiko</h3><ul>{cons}</ul></div></div></article>')
    if summary.get("watchlist"):
        rows = "".join(f'<tr><td class="l"><b>{esc(w["ticker"])}</b></td><td class="l">{esc(w["setup"])}</td><td>{w["score"]}</td>'
                       f'<td>{pc(w["probability_bullish"], False)}</td><td class="l">{esc(", ".join(w["blocked_by"]))}</td></tr>'
                       for w in summary["watchlist"])
        P.append('<h2>Watchlist</h2><div class="scroll"><table><thead><tr><th class="l">Saham</th><th class="l">Setup</th><th>Skor</th>'
                 f'<th>P(bull)</th><th class="l">Tertahan oleh</th></tr></thead><tbody>{rows}</tbody></table></div>')
    rows = "".join(f'<tr><td>{r.rank}</td><td class="l">{esc(r.sector)}</td><td>{r.score:.0f}</td><td>{pc(r.ret5)}</td><td>{pc(r.ret20)}</td>'
                   f'<td>{pc(r.ret60)}</td><td>{pc(r.rs20)}</td><td>{pc(r.breadth, False)}</td><td>{r.n_stocks}</td></tr>' for r in sectors.itertuples())
    P.append('<h2>Sector ranking</h2><div class="scroll"><table><thead><tr><th>#</th><th class="l">Sektor</th><th>Skor</th><th>5H</th><th>20H</th>'
             f'<th>60H</th><th>RS 20H</th><th>Breadth</th><th>Emiten</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if evaluation:
        P.append(f'<p class="muted">Evaluasi prediksi lama: {evaluation.get("evaluated", 0)} dievaluasi hari ini, '
                 f'{evaluation.get("pending", 0)} menunggu horizon.</p>')
    P.append(f'<footer>{esc(summary["disclaimer"])}</footer>')
    return (f'<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Daily swing report {esc(summary["date"])}</title><style>{CSS}</style></head><body><main>{"".join(P)}</main></body></html>')
