"""Komponen visual laporan HTML (chart SVG, gauge regime, CSS) — tanpa dependency.

Berisi: market overview, top opportunities (+chart entry/SL/TP), watchlist, sector strength,
backtest (equity vs pembanding), performa & kalibrasi model, prediction history.
Chart dibuat sebagai SVG inline sehingga file bisa dibuka offline di browser apa pun.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd


INK, MUTED, UP, DOWN, ACCENT, GRID, BG = "#15232D", "#66767F", "#12805C", "#C23B32", "#2E5AAC", "#D5DDE0", "#EEF2F3"
LEVEL = {"entry": "#2E5AAC", "stop": "#C23B32", "tp1": "#12805C", "tp2": "#0B5A40", "sup": "#8A6D1F", "res": "#8A6D1F"}
REGIME_ORDER = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
REGIME_LABEL = {"STRONG_BEAR": "Strong bear", "BEAR": "Bear", "NEUTRAL": "Neutral", "BULL": "Bull", "STRONG_BULL": "Strong bull"}


def esc(x) -> str:
    return html.escape(str(x))


def rp(x) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"${x:,.2f}"


def pct(x, d=1) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"


# ----------------------------------------------------------------------------- charts
def stock_chart_svg(bars: pd.DataFrame, levels: dict, w: int = 760, h: int = 420) -> str:
    """Candlestick + EMA20/50/200 + volume + RSI + garis support/resistance/entry/SL/TP."""
    b = bars.tail(120).reset_index(drop=True)
    if b.empty:
        return ""
    pad_l, pad_r = 8, 64
    ph, vh, rh, gap = 250, 60, 70, 12
    lv = [v for v in levels.values() if v is not None and np.isfinite(v)]
    lo = min(b["low"].min(), *(lv or [b["low"].min()]))
    hi = max(b["high"].max(), *(lv or [b["high"].max()]))
    lo, hi = lo - (hi - lo) * 0.03, hi + (hi - lo) * 0.03
    n = len(b)
    cw = (w - pad_l - pad_r) / n
    X = lambda i: pad_l + cw * (i + 0.5)
    Y = lambda p: 10 + (hi - p) / (hi - lo) * ph
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Chart harga" class="chart">']
    for k in range(5):
        p = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(p):.1f}" y2="{Y(p):.1f}" stroke="{GRID}" stroke-width="0.6"/>'
                   f'<text x="{w - pad_r + 6}" y="{Y(p) + 4:.1f}" class="ax">{rp(p)}</text>')
    for i, r in b.iterrows():
        col = UP if r["close"] >= r["open"] else DOWN
        out.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(r["high"]):.1f}" y2="{Y(r["low"]):.1f}" stroke="{col}" stroke-width="1"/>')
        y0, y1 = Y(max(r["open"], r["close"])), Y(min(r["open"], r["close"]))
        out.append(f'<rect x="{X(i) - cw * 0.35:.1f}" y="{y0:.1f}" width="{cw * 0.7:.1f}" height="{max(1, y1 - y0):.1f}" fill="{col}"/>')
    for col, color, dash in (("ema20", ACCENT, ""), ("ema50", "#7A5BA6", ""), ("ema200", INK, "4 3")):
        if col in b and b[col].notna().any():
            pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in b[col].items() if np.isfinite(v))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.3" stroke-dasharray="{dash}"/>')
    names = {"entry": "Entry", "stop": "SL", "tp1": "TP1", "tp2": "TP2", "sup": "Support", "res": "Resistance"}
    for k, v in levels.items():
        if v is None or not np.isfinite(v):
            continue
        dash = "2 3" if k in ("sup", "res") else "6 3"
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{LEVEL[k]}" stroke-width="1.2" stroke-dasharray="{dash}"/>'
                   f'<text x="{w - pad_r + 6}" y="{Y(v) - 3:.1f}" class="lv" fill="{LEVEL[k]}">{names[k]} {rp(v)}</text>')
    vy0 = 10 + ph + gap
    vmax = b["volume"].max() or 1
    for i, r in b.iterrows():
        hgt = r["volume"] / vmax * vh
        col = UP if r["close"] >= r["open"] else DOWN
        out.append(f'<rect x="{X(i) - cw * 0.35:.1f}" y="{vy0 + vh - hgt:.1f}" width="{cw * 0.7:.1f}" height="{hgt:.1f}" fill="{col}" opacity="0.45"/>')
    out.append(f'<text x="{w - pad_r + 6}" y="{vy0 + 12}" class="ax">Volume</text>')
    ry0 = vy0 + vh + gap
    RY = lambda v: ry0 + (100 - v) / 100 * rh
    for lvl in (30, 70):
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{RY(lvl):.1f}" y2="{RY(lvl):.1f}" stroke="{GRID}" stroke-dasharray="3 3"/>'
                   f'<text x="{w - pad_r + 6}" y="{RY(lvl) + 4:.1f}" class="ax">{lvl}</text>')
    if "rsi14" in b:
        pts = " ".join(f"{X(i):.1f},{RY(v):.1f}" for i, v in b["rsi14"].items() if np.isfinite(v))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.1"/>'
                   f'<text x="{pad_l + 2}" y="{ry0 + 10}" class="ax">RSI 14</text>')
    out.append(f'<text x="{pad_l}" y="{h - 2}" class="ax">{b["date"].iloc[0]:%d %b %Y}</text>'
               f'<text x="{w - pad_r}" y="{h - 2}" class="ax" text-anchor="end">{b["date"].iloc[-1]:%d %b %Y}</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart_svg(series: dict, w: int = 760, h: int = 260, fmt=lambda v: f"${v:,.0f}") -> str:
    series = {k: s.dropna() for k, s in series.items() if s is not None and len(s.dropna()) > 1}
    if not series:
        return "<p class='muted'>Belum ada data.</p>"
    allv = pd.concat(series.values())
    lo, hi = allv.min(), allv.max()
    if hi == lo:
        hi = lo + 1
    x0 = min(s.index.min() for s in series.values())
    x1 = max(s.index.max() for s in series.values())
    span = max((x1 - x0).days, 1)
    pl, pr, pt, pb = 8, 70, 10, 22
    X = lambda d: pl + (d - x0).days / span * (w - pl - pr)
    Y = lambda v: pt + (hi - v) / (hi - lo) * (h - pt - pb)
    colors = [ACCENT, MUTED, "#B07A1A"]
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Kurva ekuitas">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="0.6"/>'
                   f'<text x="{w - pr + 6}" y="{Y(v) + 4:.1f}" class="ax">{fmt(v)}</text>')
    legend = []
    for (name, s), col in zip(series.items(), colors):
        step = max(1, len(s) // 600)
        s2 = s.iloc[::step]
        pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in s2.items())
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{1.8 if not legend else 1.2}"/>')
        legend.append(f'<span class="key"><i style="background:{col}"></i>{esc(name)}</span>')
    out.append(f'<text x="{pl}" y="{h - 4}" class="ax">{x0:%b %Y}</text><text x="{w - pr}" y="{h - 4}" class="ax" text-anchor="end">{x1:%b %Y}</text></svg>')
    return "".join(out) + f'<div class="legend">{"".join(legend)}</div>'


def calibration_svg(table: list[dict], w: int = 360, h: int = 260) -> str:
    if not table:
        return ""
    p = 30
    S = lambda v: p + v * (w - 2 * p)
    T = lambda v: h - p - v * (h - 2 * p)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Kalibrasi">',
           f'<line x1="{S(0)}" y1="{T(0)}" x2="{S(1)}" y2="{T(1)}" stroke="{GRID}" stroke-dasharray="4 3"/>',
           f'<line x1="{S(0)}" y1="{T(0)}" x2="{S(1)}" y2="{T(0)}" stroke="{MUTED}"/><line x1="{S(0)}" y1="{T(0)}" x2="{S(0)}" y2="{T(1)}" stroke="{MUTED}"/>']
    nmax = max(r["n"] for r in table)
    pts = []
    for r in table:
        if r["predicted"] is None or r["actual"] is None:
            continue
        pts.append(f"{S(r['predicted']):.1f},{T(r['actual']):.1f}")
        out.append(f'<circle cx="{S(r["predicted"]):.1f}" cy="{T(r["actual"]):.1f}" r="{2 + 6 * np.sqrt(r["n"] / nmax):.1f}" fill="{ACCENT}" opacity="0.6"/>')
    out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ACCENT}" stroke-width="1.4"/>')
    for v in (0, 0.5, 1):
        out.append(f'<text x="{S(v)}" y="{h - 10}" class="ax" text-anchor="middle">{v:.0%}</text>'
                   f'<text x="{p - 4}" y="{T(v) + 4}" class="ax" text-anchor="end">{v:.0%}</text>')
    out.append(f'<text x="{w / 2}" y="{h}" class="ax" text-anchor="middle">Prediksi P(bullish)</text></svg>')
    return "".join(out)


def regime_gauge(regime: str) -> str:
    cells = []
    for r in REGIME_ORDER:
        on = " on" if r == regime else ""
        cells.append(f'<div class="seg{on} {r.lower()}"><span>{REGIME_LABEL[r]}</span></div>')
    return f'<div class="gauge" aria-label="Regime pasar: {REGIME_LABEL.get(regime, regime)}">{"".join(cells)}</div>'


# ----------------------------------------------------------------------------- page
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap');
:root{{--ink:{INK};--muted:{MUTED};--up:{UP};--down:{DOWN};--accent:{ACCENT};--grid:{GRID};--bg:{BG};--paper:#FFFFFF}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-variant-numeric:tabular-nums}}
main{{max-width:1080px;margin:0 auto;padding:28px 20px 80px}}
h1{{font:600 34px/1.1 "IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;margin:0 0 6px;letter-spacing:-.01em}}
h2{{font:600 21px/1.2 "IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;margin:44px 0 12px;padding-top:14px;border-top:2px solid var(--ink)}}
h3{{font-size:16px;margin:0 0 4px}}
p{{max-width:72ch}}
.muted{{color:var(--muted)}}
.warn{{background:#FBEAEA;border-left:4px solid var(--down);padding:10px 14px;margin:14px 0;max-width:none}}
.gauge{{display:grid;grid-template-columns:repeat(5,1fr);gap:3px;margin:18px 0 8px}}
.seg{{height:44px;display:flex;align-items:flex-end;padding:6px 8px;font-size:13px;color:var(--muted);background:#E1E7EA}}
.seg.on{{color:#fff;font-weight:600}}
.seg.on.strong_bear{{background:#8E2A24}}.seg.on.bear{{background:var(--down)}}.seg.on.neutral{{background:#5E6E78}}
.seg.on.bull{{background:var(--up)}}.seg.on.strong_bull{{background:#0B5A40}}
.facts{{display:flex;flex-wrap:wrap;gap:6px 28px;margin:6px 0 0;font-size:14px}}
.facts b{{font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--paper)}}
th,td{{padding:7px 9px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}}
th{{font-weight:600;color:var(--muted);background:#F6F8F9;position:sticky;top:0}}
td:first-child,th:first-child,td.l,th.l{{text-align:left}}
.scroll{{overflow-x:auto;max-width:100%}}
.dec{{display:inline-block;padding:1px 8px;font-size:12px;font-weight:600;border:1px solid}}
.dec.BUY{{color:var(--up);border-color:var(--up)}}.dec.WATCHLIST{{color:var(--accent);border-color:var(--accent)}}
.dec.WAIT{{color:var(--muted);border-color:var(--grid)}}.dec.AVOID{{color:var(--down);border-color:var(--down)}}
.pick{{background:var(--paper);margin:18px 0;padding:18px 18px 10px;border-left:4px solid var(--up)}}
.pick header{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline}}
.pick .score{{font:600 28px "IBM Plex Sans Condensed",sans-serif}}
.levels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:12px 0}}
.levels div{{font-size:13px;color:var(--muted)}}.levels b{{display:block;color:var(--ink);font-size:16px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
ul{{margin:4px 0 10px;padding-left:18px}} li{{margin:2px 0}}
li.pro::marker{{content:"+  ";color:var(--up)}} li.con::marker{{content:"–  ";color:var(--down)}}
.chart{{width:100%;height:auto;display:block;background:var(--paper)}}
.ax{{font-size:10.5px;fill:var(--muted)}} .lv{{font-size:10.5px;font-weight:600}}
.legend{{display:flex;gap:18px;font-size:13px;margin:6px 0}} .key i{{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle}}
.empty{{background:var(--paper);padding:22px;font-size:17px;border-left:4px solid var(--muted)}}
.pos{{color:var(--up)}} .neg{{color:var(--down)}}
footer{{margin-top:50px;font-size:13px;color:var(--muted)}}
@media (max-width:720px){{.cols{{grid-template-columns:1fr}} h1{{font-size:27px}} .seg span{{font-size:11px}}}}
"""


def _signed(v, d=1, suffix="%"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span class="{cls}">{v * 100:+.{d}f}{suffix}</span>'


