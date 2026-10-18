"""Dashboard production (§44, §45) — membaca database cloud (Aiven PostgreSQL), bukan file lokal.

Deploy sebagai WEB SERVICE terpisah dari daily job (Streamlit Community Cloud / Render / Docker).
Credential: DATABASE_URL dari st.secrets (Streamlit Cloud) atau environment. Dashboard hanya membaca.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

try:  # Streamlit Cloud: secrets → environment (sebelum config dimuat)
    for k in ("DATABASE_URL", "APP_ENV"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

from app.config import get, load_config  # noqa: E402
from app.pipeline.context import AppContext  # noqa: E402
from app.reporting import dashboard_helpers as H  # noqa: E402

UP, DOWN, ACCENT, INK = "#12805C", "#C23B32", "#2E5AAC", "#15232D"
st.set_page_config(page_title="Swing Stock Predictor — US", layout="wide")


@st.cache_resource
def ctx() -> AppContext:
    return AppContext.create(load_config(ROOT / "config"), migrate_db=False)


@st.cache_data(ttl=300)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return ctx().db.query_df(sql, params)


@st.cache_data(ttl=600)
def stock_frame(ticker: str, days: int = 400) -> pd.DataFrame:
    from app.features.technical import compute_stock_features
    df = q("SELECT p.date, p.open, p.high, p.low, p.close, p.volume, p.value FROM price_history p JOIN stocks s ON s.id = p.stock_id "
           "WHERE s.ticker = ? ORDER BY p.date DESC LIMIT ?", (ticker, days))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").assign(ticker=ticker)
    for c in ("open", "high", "low", "close", "volume", "value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return compute_stock_features(df, ctx().cfg)


def jl(v):
    try:
        return json.loads(v) if isinstance(v, str) else (v or [])
    except Exception:
        return []


def _connect_check():
    c = ctx()
    try:
        c.db.scalar("SELECT 1")
    except Exception:
        # koneksi lama rusak (transaksi aborted / ditutup pooler) → buang & sambung ulang sekali
        try:
            c.db.close()
        except Exception:
            pass
        c.db.scalar("SELECT 1")
    return c


try:
    c = _connect_check()
except Exception as e:
    st.cache_resource.clear()
    st.error(f"Database tidak bisa diakses: {type(e).__name__}. Periksa DATABASE_URL di secrets "
             "(Service URI Aiven, sebaiknya role dashboard_ro dari scripts/create_dashboard_role.py), lalu Reboot app.")
    st.stop()


@st.cache_data(ttl=300)
def research_ready() -> str | None:
    """None bila tabel/kolom riset sudah ada; selain itu pesan. Skema dibuat/diperbarui oleh workflow Daily/Setup."""
    for sql in ("SELECT 1 FROM valuation_results LIMIT 1", "SELECT 1 FROM accumulation_signals LIMIT 1",
                "SELECT 1 FROM foreign_flow_history LIMIT 1", "SELECT value_score, accumulation_status FROM predictions LIMIT 1",
                "SELECT category FROM data_sources LIMIT 1"):
        try:
            c.db.query(sql)
        except Exception:
            return ("Skema database belum memuat tabel riset (valuasi / short flow / akumulasi). Jalankan workflow "
                    "**Daily Market Pipeline** sekali (Actions → Run workflow) — migrasi berjalan otomatis — lalu muat ulang.")
    return None

if hasattr(H, "stale_code_warning"):
    _stale = H.stale_code_warning(H.git_commit())
else:  # modul helper sendiri masih versi lama (dimuat sebelum update ini) → pasti perlu reboot
    _stale = "Versi kode baru terdeteksi, tetapi aplikasi masih memakai modul lama. Klik **Manage app → ⋮ → Reboot app**."
if _stale:
    st.warning(_stale)

_banner = H.database_banner(c.db)
if _banner:
    (st.error if _banner[0] == "error" else st.warning)(_banner[1])
    if _banner[0] == "error":
        st.stop()

PAGES = ["Market Overview", "Full-Market Coverage", "Top Swing Stocks", "Sector Strength", "Stock Detail", "Undervalued Screener", "Short Flow (FINRA)",
         "Accumulation / Distribution", "Integrated Research", "Prediction History", "Backtest", "Model Performance", "Model Lab",
         "Pipeline Status", "Data Health"]
page = st.sidebar.radio("Halaman", PAGES)
st.sidebar.caption("Analytical decision support — bukan jaminan profit. Bukan nasihat investasi.")

last_sig_date = c.db.scalar("SELECT MAX(signal_date) FROM trading_signals") or c.db.scalar("SELECT MAX(prediction_date) FROM predictions")
last_sig_date = str(last_sig_date)[:10] if last_sig_date else None
st.sidebar.caption(H.connection_caption(c.db, last_sig_date))


@st.cache_data(ttl=600)
def latest_test_backtest():
    """Run walk-forward TEST terbaru (out-of-sample) → (metrics, benchmark) untuk peringatan kinerja strategi."""
    r = q("SELECT metrics, benchmark FROM backtest_runs WHERE run_id LIKE ? ORDER BY created_at DESC LIMIT 1", ("wf-test-%",))
    return (jl(r["metrics"].iloc[0]), jl(r["benchmark"].iloc[0])) if len(r) else (None, None)


def _safe_q(sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        return q(sql, params)
    except Exception:                       # tabel Alpha Vantage belum ada (sebelum migrasi v9)
        return pd.DataFrame()


def show_earnings_and_news(tk: str) -> None:
    """Earnings & berita dari Alpha Vantage: rilis berikutnya (resmi), riwayat kejutan EPS, berita + sentimen."""
    st.subheader("Earnings & berita")
    cal = _safe_q("SELECT c.fiscal_date_ending, c.report_date, c.estimate, c.time_of_day, c.as_of FROM earnings_calendar c "
                  "JOIN stocks s ON s.id = c.stock_id WHERE s.ticker = ? ORDER BY c.as_of DESC", (tk,))
    sur = _safe_q("SELECT e.fiscal_date_ending, e.reported_date, e.report_time, e.reported_eps, e.estimated_eps, "
                  "e.surprise, e.surprise_pct FROM earnings_surprises e JOIN stocks s ON s.id = e.stock_id "
                  "WHERE s.ticker = ? ORDER BY e.fiscal_date_ending DESC LIMIT 8", (tk,))
    k = st.columns(3)
    if len(cal):
        cal = cal.drop_duplicates("fiscal_date_ending", keep="first")
        up = cal[pd.to_datetime(cal["report_date"]) >= pd.Timestamp.today().normalize()].sort_values("report_date")
        if len(up):
            r = up.iloc[0]
            days = (pd.Timestamp(r["report_date"]) - pd.Timestamp.today().normalize()).days
            k[0].metric("Rilis laba berikutnya", str(r["report_date"])[:10], f"{days} hari lagi", delta_color="off")
            est = pd.to_numeric(r["estimate"], errors="coerce")
            k[1].metric("Konsensus EPS", f"${est:,.2f}" if pd.notna(est) else "n/a", r["time_of_day"] or None,
                        delta_color="off")
    if len(sur):
        beat = (pd.to_numeric(sur["surprise"], errors="coerce") > 0).mean()
        k[2].metric("Beat rate 8 kuartal", f"{beat:.0%}")
        show = sur.rename(columns={"fiscal_date_ending": "kuartal", "reported_date": "rilis", "report_time": "jam",
                                   "reported_eps": "EPS aktual", "estimated_eps": "EPS estimasi", "surprise": "surprise",
                                   "surprise_pct": "surprise %"})
        st.dataframe(show, hide_index=True, use_container_width=True)
    elif not len(cal):
        st.caption("Belum ada data earnings Alpha Vantage untuk saham ini (backfill ±20 saham/hari; atau secret "
                   "ALPHAVANTAGE_API_KEY belum diset).")
    news = _safe_q("SELECT n.published_at, n.headline, n.source, n.sentiment, n.relevance FROM news_items n "
                   "JOIN stocks s ON s.id = n.stock_id WHERE s.ticker = ? ORDER BY n.published_at DESC LIMIT 12", (tk,))
    if len(news):
        avg = pd.to_numeric(news["sentiment"], errors="coerce").mean()
        st.caption(f"Sentimen berita rata-rata {avg:+.2f} (−1 sangat negatif … +1 sangat positif; ≥ ±0,15 = condong)")
        st.dataframe(news.rename(columns={"published_at": "waktu (ET)", "headline": "judul", "source": "sumber",
                                          "sentiment": "sentimen", "relevance": "relevansi"}),
                     hide_index=True, use_container_width=True)


def show_macro():
    m = _safe_q("SELECT series, date, value FROM macro_series WHERE series IN ('UST10YEAR', 'UST2YEAR', 'UST3MONTH') "
                "AND date >= ? ORDER BY date", ((pd.Timestamp.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),))
    if m.empty:
        return
    w = m.pivot_table(index="date", columns="series", values="value").sort_index()
    last = w.ffill().iloc[-1]
    k = st.columns(4)
    for i, (lab, col) in enumerate((("Treasury 10Y", "UST10YEAR"), ("Treasury 2Y", "UST2YEAR"), ("Treasury 3M", "UST3MONTH"))):
        if col in last and pd.notna(last[col]):
            k[i].metric(lab, f"{last[col]:.2f}%")
    if {"UST10YEAR", "UST3MONTH"} <= set(last.index):
        slope = last["UST10YEAR"] - last["UST3MONTH"]
        k[3].metric("Kurva 10Y − 3M", f"{slope:+.2f} pp", "terbalik (inversi)" if slope < 0 else "normal",
                    delta_color="inverse" if slope < 0 else "off")
    st.caption(f"Yield Treasury AS (Alpha Vantage) per {str(w.index[-1])[:10]}; 10Y dipakai sebagai risk-free valuasi.")


def show_strategy_verdict():
    v = H.strategy_verdict(*latest_test_backtest())
    if v:
        (st.warning if v[0] == "warning" else st.info)(v[1])

if page == "Market Overview":
    ix = q("SELECT date, close FROM market_index WHERE symbol = ? ORDER BY date DESC LIMIT 500", (get(c.cfg, "data.index_id"),))
    run = q("SELECT run_id, status, started_at, data_status, market_date, summary FROM pipeline_runs WHERE run_type = 'daily' "
            "ORDER BY started_at DESC LIMIT 1")
    s = jl(run["summary"].iloc[0]) if len(run) and run["summary"].iloc[0] else {}
    st.title(f"Pasar {last_sig_date or '–'}: {s.get('regime', 'n/a')}")
    st.subheader(s.get("headline", "–"))
    k = st.columns(4)
    k[0].metric("Status pipeline", run["status"].iloc[0] if len(run) else "–")
    k[1].metric("Data", (run["data_status"].iloc[0] or "–").replace("_", " ").title() if len(run) else "–")
    k[2].metric("Rekomendasi BUY", len(s.get("top", [])))
    cs = s.get("coverage") or {}
    k[3].metric("Layak sinyal", f"{cs['signal_eligible']}/{cs['total_universe']}" if cs.get("total_universe") else "–")
    if len(run):
        st.caption(f"Run {run['run_id'].iloc[0]} · mulai {str(run['started_at'].iloc[0])[:16]} UTC")
    show_strategy_verdict()
    show_macro()
    if len(ix):
        ix = ix.sort_values("date")
        st.plotly_chart(go.Figure(go.Scatter(x=pd.to_datetime(ix["date"]), y=pd.to_numeric(ix["close"]), line=dict(color=INK)))
                        .update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), title="S&P 500"), use_container_width=True)
    rep = c.reports.latest("html")
    if rep:
        with st.expander(f"Daily report terbaru ({rep[0]})"):
            st.components.v1.html(rep[1].decode("utf-8"), height=900, scrolling=True)

elif page == "Full-Market Coverage":
    from app.scanner.coverage import STATUSES
    try:
        cov = c.repo.coverage()
    except Exception as e:
        st.error(f"Tabel `scan_coverage` tidak bisa dibaca ({type(e).__name__}). Pastikan dashboard tersambung ke "
                 "database Aiven dan role dashboard punya izin SELECT (jalankan ulang scripts/create_dashboard_role.py).")
        st.stop()
    if cov.empty:
        st.info("Belum ada data cakupan. Tabel `scan_coverage` diisi oleh workflow Daily Market Pipeline.")
        st.stop()
    d = cov["scan_date"].iloc[0]
    st.title(f"Cakupan pemindaian seluruh universe — {d}")
    n = len(cov)
    counts = cov["status"].value_counts()
    k = st.columns(4)
    k[0].metric("Universe aktif", n)
    k[1].metric("Layak sinyal (SUCCESS)", f"{int(counts.get('SUCCESS', 0))} ({counts.get('SUCCESS', 0) / n:.0%})")
    k[2].metric("Gagal provider", int(counts.get("PROVIDER_ERROR", 0)))
    k[3].metric("Dikecualikan", int(counts.get("EXCLUDED", 0)))
    st.caption("Hanya status SUCCESS yang boleh menjadi BUY. Status lain tetap tercatat beserta alasannya. "
               f"Diperbarui oleh run {cov['run_id'].iloc[0]}.")
    st.dataframe(pd.DataFrame({"status": STATUSES, "jumlah": [int(counts.get(s_, 0)) for s_ in STATUSES]}),
                 hide_index=True)
    pick = st.multiselect("Filter status", STATUSES, default=[s_ for s_ in STATUSES if s_ != "SUCCESS"])
    view = cov[cov["status"].isin(pick)]
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button("Unduh CSV", view.to_csv(index=False).encode("utf-8"), f"scan_coverage_{d}.csv", "text/csv")

elif page == "Top Swing Stocks":
    st.title(f"Top swing stocks — {last_sig_date or '–'}")
    show_strategy_verdict()
    sig = q("SELECT * FROM trading_signals WHERE signal_date = ? ORDER BY rank", (last_sig_date,)) if last_sig_date else pd.DataFrame()
    det = q("SELECT ticker, setup, score, prob_bullish, market_regime, reasons, risks FROM predictions "
            "WHERE prediction_date = ? AND decision = 'BUY' ORDER BY score DESC", (last_sig_date,)) if last_sig_date else pd.DataFrame()
    inside, outside = H.split_buys(det, sig)

    def _detail(r):
        with st.expander(r.ticker):
            a, b = st.columns(2)
            a.markdown("**Alasan**\n" + "\n".join(f"- {x}" for x in jl(r.reasons)))
            b.markdown("**Risiko**\n" + ("\n".join(f"- {x}" for x in jl(r.risks)) or "- tidak ada yang menonjol"))

    if sig.empty:
        st.subheader("NO HIGH-CONVICTION SETUP TODAY")
        st.write("Tidak ada saham yang lolos semua filter risiko. Menunggu adalah posisi yang sah.")
    else:
        st.dataframe(sig[["rank", "ticker", "setup", "score", "prob_bullish", "expected_return", "entry_low", "entry_ideal", "entry_high",
                          "stop_loss", "tp1", "tp2", "risk_reward", "lots", "confidence", "decision"]], hide_index=True, use_container_width=True)
        for r in inside.itertuples():
            _detail(r)
    if len(outside):
        regime = outside["market_regime"].iloc[0]
        cap = get(c.cfg, f"regime.rules.{regime}.max_recommendations")
        st.subheader("BUY di luar ranking")
        st.caption(f"Lolos semua filter, tetapi tidak direkomendasikan karena regime {regime} membatasi rekomendasi "
                   f"maksimal {cap} saham per hari. Bukan sinyal untuk dieksekusi.")
        st.dataframe(outside[["ticker", "setup", "score", "prob_bullish"]], hide_index=True, use_container_width=True)
        for r in outside.itertuples():
            _detail(r)
    w = q("SELECT ticker, setup, score, prob_bullish, reject_reasons FROM predictions WHERE prediction_date = ? AND decision = 'WATCHLIST' "
          "ORDER BY score DESC LIMIT 15", (last_sig_date,)) if last_sig_date else pd.DataFrame()
    if len(w):
        st.subheader("Watchlist")
        st.dataframe(w.assign(reject_reasons=w["reject_reasons"].map(lambda v: ", ".join(jl(v)))), hide_index=True, use_container_width=True)

elif page == "Sector Strength":
    d = c.db.scalar("SELECT MAX(date) FROM sector_data")
    sec = q("SELECT * FROM sector_data WHERE date = ? ORDER BY rank", (d,)) if d else pd.DataFrame()
    st.title(f"Kekuatan sektor — {d or '–'}")
    if len(sec):
        st.plotly_chart(go.Figure(go.Bar(x=pd.to_numeric(sec["score"]), y=sec["sector"], orientation="h", marker_color=ACCENT))
                        .update_layout(height=420, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10)),
                        use_container_width=True)
        st.dataframe(sec.drop(columns=["id"]), hide_index=True, use_container_width=True)

elif page == "Stock Detail":
    tickers = q("SELECT ticker FROM stocks WHERE is_active = ? ORDER BY ticker", (True,))["ticker"].tolist()
    if not tickers:
        st.stop()
    tk = st.selectbox("Saham", tickers)
    g = stock_frame(tk)
    p = q("SELECT * FROM predictions WHERE ticker = ? ORDER BY prediction_date DESC LIMIT 1", (tk,))
    if g.empty:
        st.warning("Belum ada data harga.")
        st.stop()
    g = g.tail(160)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.18, 0.22], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name=tk,
                                 increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    for col, color in (("ema20", ACCENT), ("ema50", "#7A5BA6"), ("ema200", INK)):
        fig.add_trace(go.Scatter(x=g["date"], y=g[col], name=col.upper(), line=dict(color=color, width=1.2)), 1, 1)
    if len(p):
        r = p.iloc[0]
        for name, col, color in (("Entry", "entry_ideal", ACCENT), ("SL", "stop_loss", DOWN), ("TP1", "tp1", UP), ("TP2", "tp2", "#0B5A40")):
            v = pd.to_numeric(r[col], errors="coerce")
            if pd.notna(v):
                fig.add_hline(y=float(v), line=dict(color=color, dash="dash", width=1), annotation_text=f"{name} ${v:,.2f}", row=1, col=1)
    for name, col in (("Support", "sup20"), ("Resistance", "res50")):
        v = g[col].iloc[-1]
        if pd.notna(v):
            fig.add_hline(y=float(v), line=dict(color="#8A6D1F", dash="dot", width=1), annotation_text=name, row=1, col=1)
    fig.add_trace(go.Bar(x=g["date"], y=g["volume"], marker_color=np.where(g["close"] >= g["open"], UP, DOWN), name="Volume"), 2, 1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], line=dict(color=INK, width=1), name="RSI14"), 3, 1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    # tanpa celah akhir pekan / libur bursa / hari suspensi
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=H.missing_business_days(g["date"]))])
    st.plotly_chart(fig, use_container_width=True)
    if len(p):
        r = p.iloc[0]
        k = st.columns(4)
        k[0].metric("Keputusan", r["decision"])
        k[1].metric("P(bullish)", f"{float(r['prob_bullish']):.0%}")
        k[2].metric("Risk/reward", f"{float(r['risk_reward']):.2f}" if pd.notna(r["risk_reward"]) else "n/a")
        k[3].metric("Model", r["model_version"])
        st.caption(f"Prediksi {r['prediction_date']} · data {r['data_status']}")
    show_earnings_and_news(tk)

elif page == "Prediction History":
    st.title("Riwayat prediksi")
    dec = st.multiselect("Keputusan", ["BUY", "WATCHLIST", "WAIT", "AVOID"], default=["BUY", "WATCHLIST"])
    h = c.repo.prediction_history(20000)
    h = h[h["decision"].isin(dec)]
    if len(h):
        h = h.assign(prediction_date=pd.to_datetime(h["prediction_date"]).dt.strftime("%Y-%m-%d"))
    ev = h[h["outcome"].notna()]
    if len(ev):
        k = st.columns(4)
        k[0].metric("Terevaluasi", len(ev))
        k[1].metric("Prediksi benar", f"{ev['prediction_correct'].astype(float).mean():.0%}")
        k[2].metric("Kena TP1", f"{ev['hit_tp1'].astype(float).mean():.0%}")
        k[3].metric("Kena SL", f"{ev['hit_stop'].astype(float).mean():.0%}")
    st.dataframe(h, hide_index=True, use_container_width=True)

elif page == "Backtest":
    st.title("Backtest")
    runs = c.repo.backtest_runs(30)
    if runs.empty:
        st.info("Belum ada backtest (workflow Weekly Backtest).")
        st.stop()
    st.caption("wf-test = out-of-sample terbaru (paling relevan) · wf-validation = periode validasi · "
               "live = sinyal produksi · modules = evaluasi modul riset per varian. Semua angka setelah biaya.")
    rid = st.selectbox("Run", runs["run_id"], index=H.default_backtest_run(runs))
    r = runs[runs["run_id"] == rid].iloc[0]
    m, b = jl(r["metrics"]), jl(r["benchmark"])
    kpi = H.backtest_kpis(m)
    if kpi:
        b = b if isinstance(b, dict) else {}
        k = st.columns(5)
        k[0].metric("Total return", f"{kpi['total_return']:+.1%}")
        k[1].metric("Max drawdown", f"{kpi['max_drawdown']:.1%}")
        k[2].metric("Sharpe", f"{kpi['sharpe']:.2f}")
        k[3].metric("Trades", kpi["number_of_trades"])
        bh = b.get("ihsg_buy_hold") or {}
        k[4].metric("S&P 500 buy & hold", f"{bh['total_return']:+.1%}" if bh.get("total_return") is not None else "n/a")
        if b.get("technical_only", {}).get("total_return") is not None:
            st.caption(f"Pembanding teknikal-saja (tanpa ML): {b['technical_only']['total_return']:+.1%}, "
                       f"Sharpe {(b['technical_only'].get('sharpe') or float('nan')):.2f}")
        sig = m.get("significance") if isinstance(m, dict) else None
        if sig:
            st.subheader("Bukti statistik (apakah ini skill atau keberuntungan?)")
            s = st.columns(5)
            s[0].metric("Alpha / tahun", f"{sig['alpha_annual']:+.1%}" if sig.get("alpha_annual") is not None else "n/a",
                        help="Return di luar eksposur pasar (regresi harian terhadap S&P 500)")
            s[1].metric("t-stat alpha", sig.get("alpha_tstat") if sig.get("alpha_tstat") is not None else "n/a",
                        help="|t| ≥ 2 ≈ signifikan")
            s[2].metric("Beta", sig.get("beta") if sig.get("beta") is not None else "n/a")
            s[3].metric("Sharpe 90% CI", f"[{sig.get('sharpe_ci_low')}, {sig.get('sharpe_ci_high')}]"
                        if sig.get("sharpe_ci_low") is not None else "n/a")
            s[4].metric("Deflated Sharpe", sig.get("dsr") if sig.get("dsr") is not None else "n/a",
                        help="Probabilitas Sharpe > keberuntungan setelah memperhitungkan jumlah konfigurasi yang dicoba")
            if sig.get("beta_matched_index_return") is not None:
                st.caption(f"Pembanding ADIL = beta × S&P 500 (eksposur pasar yang sama): "
                           f"{sig['beta_matched_index_return']:+.1%}. {sig.get('verdict', '')}")
    elif isinstance(m, dict) and m.get("periods"):
        st.subheader("Perbandingan varian (setelah biaya)")
        vt = H.module_variant_table(m)
        st.dataframe(vt, hide_index=True, use_container_width=True)
    with st.expander("Detail lengkap (JSON)"):
        st.json({"strategy": m, "benchmark": b})
    t = q("SELECT * FROM backtest_trades WHERE backtest_run_id = ? ORDER BY entry_date", (rid,))
    st.dataframe(t, hide_index=True, use_container_width=True)

elif page == "Model Performance":
    st.title("Model")
    v = c.models.list_versions()
    st.dataframe(v.drop(columns=["metrics"]), hide_index=True, use_container_width=True)
    if len(v):
        ver = st.selectbox("Versi", v["version"])
        m = jl(v[v["version"] == ver]["metrics"].iloc[0])
        if isinstance(m, dict) and m.get("folds"):
            st.subheader("Walk-forward per fold")
            st.dataframe(pd.DataFrame(m["folds"]), hide_index=True, use_container_width=True)
            st.json({k: m.get(k) for k in ("auc_mean", "auc_std", "rank_ic_mean", "ece_bullish", "positive_fold_ratio",
                                            "head_to_head", "test_sealed")})
    st.subheader("Riset faktor — rank IC vs return residual 5 hari")
    st.caption("Korelasi peringkat harian setiap fitur dengan return saham dikurangi beta × S&P 500, hanya tanggal yang "
               "tidak tumpang tindih. |t| ≥ 2 ≈ signifikan. IC 12 bulan menunjukkan apakah sinyal masih hidup.")
    fr = c.reports.latest("factor_ic")
    if fr:
        import io
        st.dataframe(pd.read_csv(io.BytesIO(fr[1])), hide_index=True, use_container_width=True)
    else:
        st.info("Belum ada laporan faktor — dibuat oleh workflow Weekly Backtest (atau `python main.py factor-report`).")

elif page == "Model Lab":
    from app.challengers import MODEL_LABELS, challenger_cfg
    from app.challengers.report import comparison_table, live_evaluation
    st.title("Model Lab — LSTM · GRU · ARIMA-GARCH · TFT")
    st.caption("Model challenger berjalan TERPISAH: tidak memengaruhi keputusan BUY/WAIT. Metrik di bawah dihitung pada "
               "periode test tersegel dengan label, biaya, dan definisi yang sama dengan model aktif.")

    @st.cache_data(ttl=600)
    def _table():
        return comparison_table(ctx().repo)

    try:
        t = _table()
    except Exception as e:
        if isinstance(e, AttributeError) or _stale:
            st.error("Modul aplikasi di memori masih versi lama. Klik **Manage app → ⋮ → Reboot app**.")
        else:
            st.error(f"Tabel challenger belum bisa dibaca ({type(e).__name__}: {str(e)[:200]}). "
                     "Bila workflow Challenger Train belum pernah sukses, jalankan sekali.")
        st.stop()
    if t.empty:
        st.info("Belum ada model challenger terlatih. Jalankan Actions → Challenger Train → Run workflow.")
        st.stop()
    st.subheader("Perbandingan out-of-sample")
    st.dataframe(t.style.format({c_: "{:.3f}" for c_ in ("AUC bullish", "log loss", "ECE", "ECE mentah", "IC harian", "IC t",
                                                        "AUC naik", "cakupan 80%", "p top-k aturan")}
                                | {"top-k net": "{:+.2%}", "top-k aturan": "{:+.2%}"}, na_rep="–"),
                 hide_index=True, use_container_width=True)
    st.caption("top-k net = rata-rata return 5 hari top-5 saham per hari (urut P(bullish), definisi model aktif) setelah biaya. "
               "Aturan ranking & kalibrasi dipilih di fold VALIDASI, lalu dilaporkan di test. CI95/p dari bootstrap blok "
               "5 hari per tanggal: bila CI memuat 0, hasil belum terbukti. ECE ≤ 0,06 = syarat kalibrasi. Cakupan 80% ideal ≈ 0,80.")
    reg = q("SELECT model_name, version, created_at, metrics FROM challenger_models ORDER BY created_at DESC")
    for name in reg["model_name"].unique():
        r = reg[reg["model_name"] == name].iloc[0]
        m = jl(r["metrics"])
        with st.expander(f"{MODEL_LABELS.get(name, name)} {r['version']} — kesimpulan"):
            for line in (m.get("verdict") or []):
                st.markdown(f"- {line}")
            st.caption(f"Dilatih {str(r['created_at'])[:16]} · waktu {m.get('seconds_total', '–')} detik")

    d = c.db.scalar("SELECT MAX(forecast_date) FROM model_forecasts")
    if d:
        d = str(d)[:10]
        st.subheader(f"Prakiraan terbaru — {d}")
        try:
            f = q("SELECT ticker, model_name, rank_score, exp_return, prob_up, prob_bullish, prob_bearish, vol, q10, q90 "
                  "FROM model_forecasts WHERE forecast_date = ?", (d,))
        except Exception:  # kolom rank_score belum dimigrasi (workflow berikutnya menambahkannya)
            f = q("SELECT ticker, model_name, exp_return, prob_up, prob_bullish, prob_bearish, vol, q10, q90 "
                  "FROM model_forecasts WHERE forecast_date = ?", (d,))
        mdl = st.selectbox("Model", sorted(f["model_name"].unique()), format_func=lambda x: MODEL_LABELS.get(x, x))
        sort_col = "rank_score" if "rank_score" in f and f.loc[f["model_name"] == mdl, "rank_score"].notna().any() else "prob_bullish"
        top = f[f["model_name"] == mdl].sort_values(sort_col, ascending=False).head(25)
        st.caption(f"Diurutkan berdasarkan {'aturan ranking terpilih model (rank_score)' if sort_col == 'rank_score' else 'P(bullish)'}.")
        st.dataframe(top.drop(columns=["model_name"]), hide_index=True, use_container_width=True)
        tk = st.selectbox("Bandingkan semua model untuk satu saham", sorted(f["ticker"].unique()))
        cmp_ = f[f["ticker"] == tk].assign(model=lambda x: x["model_name"].map(MODEL_LABELS))
        act = q("SELECT prob_bullish, expected_return FROM predictions WHERE ticker = ? ORDER BY prediction_date DESC LIMIT 1", (tk,))
        if len(act):
            cmp_ = pd.concat([cmp_, pd.DataFrame([{"model": "Model aktif", "prob_bullish": act["prob_bullish"].iloc[0],
                                                    "exp_return": act["expected_return"].iloc[0]}])])
        st.dataframe(cmp_[["model", "exp_return", "prob_up", "prob_bullish", "prob_bearish", "vol", "q10", "q90"]],
                     hide_index=True, use_container_width=True)

        st.subheader("Evaluasi live (prakiraan yang horizonnya sudah lewat)")
        cc = challenger_cfg(c.cfg)
        since = (pd.Timestamp(d) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
        fc_all = q("SELECT forecast_date, ticker, model_name, exp_return, prob_bullish, q10, q90 FROM model_forecasts "
                   "WHERE forecast_date >= ?", (since,))
        if len(fc_all):
            fc_all["forecast_date"] = pd.to_datetime(fc_all["forecast_date"])
            px = q("SELECT s.ticker, p.date, p.close FROM price_history p JOIN stocks s ON s.id = p.stock_id WHERE p.date >= ?",
                   (since,))
            px["date"] = pd.to_datetime(px["date"])
            px["close"] = pd.to_numeric(px["close"])
            live = live_evaluation(fc_all, px, cc["horizon"], cc["bull_threshold"])
            if live.empty:
                st.info(f"Belum ada prakiraan yang berumur ≥ {cc['horizon']} hari bursa.")
            else:
                st.dataframe(live, hide_index=True, use_container_width=True)
                st.caption("Out-of-sample sungguhan: prakiraan dibuat sebelum hasilnya diketahui. Butuh puluhan hari "
                           "sebelum angka ini bermakna.")
    else:
        st.info("Belum ada prakiraan harian. Workflow Challenger Predict berjalan otomatis setelah Daily Market Pipeline.")

elif page == "Pipeline Status":
    st.title("Pipeline runs")
    runs = c.repo.pipeline_runs(100)
    st.dataframe(runs.drop(columns=["summary"], errors="ignore"), hide_index=True, use_container_width=True)
    logs = q("SELECT ts, level, logger, message, run_id FROM system_logs ORDER BY ts DESC LIMIT 200")
    st.subheader("Log penting")
    st.dataframe(logs, hide_index=True, use_container_width=True)

elif page == "Data Health":
    from app.pipeline.health import health_check
    st.title("Data health")
    h = health_check(c, check_provider=False)
    st.dataframe(pd.DataFrame([{"check": k, **v} for k, v in h["checks"].items()]), hide_index=True, use_container_width=True)
    st.json(c.repo.price_counts())
    st.subheader("Sumber data")
    st.dataframe(q("SELECT name, type, status, detail, checked_at, last_success_at, last_error_at, last_error, rows_fetched_total FROM data_sources ORDER BY type, name"),
                 hide_index=True, use_container_width=True)


# ================================================================ halaman riset (logika data: app/research/dashboard_data.py)
elif page == "Undervalued Screener":
    from app.research import dashboard_data as dd
    if research_ready():
        st.warning(research_ready())
        st.stop()
    st.title("Undervalued screener")
    st.caption("Nilai wajar = RENTANG estimasi berbasis asumsi (config/research.yaml). Keputusan VALUE terpisah dari SWING. "
               "Bukan nasihat investasi.")
    cov = dd.valuation_coverage(c.repo)
    k = st.columns(3)
    k[0].metric("Tanggal valuasi", cov.get("date") or "–")
    k[1].metric("Emiten dinilai", sum(v for s_, v in cov.get("by_status", {}).items() if s_ != "INSUFFICIENT_DATA"))
    k[2].metric("INSUFFICIENT_DATA", cov.get("by_status", {}).get("INSUFFICIENT_DATA", 0))
    a, b, c3, d4 = st.columns(4)
    min_mos = a.slider("Margin of safety minimum", -0.5, 0.8, 0.2, 0.05)
    statuses = b.multiselect("Status", ["DEEP_VALUE", "UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"], ["DEEP_VALUE", "UNDERVALUED"])
    excl = c3.checkbox("Sembunyikan value trap HIGH", True)
    minq = d4.slider("Quality score minimum", 0, 100, 0, 5)
    tbl = dd.undervalued_screener(c.repo, min_mos, statuses, excl, None, minq or None)
    if tbl.empty:
        st.info("Tidak ada emiten yang memenuhi filter, atau belum ada valuasi. Valuasi butuh laporan keuangan — "
                "lihat status sumber di bawah.")
    else:
        st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.subheader("Status sumber fundamental & short flow")
    st.dataframe(dd.source_status(c.repo), hide_index=True, use_container_width=True)

elif page == "Short Flow (FINRA)":
    from app.research import dashboard_data as dd
    if research_ready():
        st.warning(research_ready())
        st.stop()
    st.title("Short flow — FINRA daily short sale volume")
    st.caption("Padanan AS untuk modul foreign flow versi BEI. NET_BUYING = rasio short di BAWAH kebiasaan 120 hari "
               "saham itu sendiri (tekanan jual-short mereda); NET_SELLING = di atasnya. Volume FINRA = volume "
               "off-exchange yang dilaporkan ke FINRA, bukan volume konsolidasi.")
    ov = dd.foreign_flow_overview(c.repo)
    if ov["status"] != "AVAILABLE":
        st.warning("FOREIGN_FLOW_UNAVAILABLE — belum ada data FINRA. Data diunduh otomatis oleh Daily/Setup dari "
                   "cdn.finra.org; cek tabel status sumber di bawah. Skor & ranking flow tidak dihitung (bukan nol).")
        st.dataframe(dd.source_status(c.repo), hide_index=True, use_container_width=True)
    else:
        dly = ov["daily"]
        st.caption(f"Data terakhir {ov['last_date']}.")
        st.plotly_chart(go.Figure(go.Scatter(x=dly["date"], y=dly["short_ratio"] * 100, mode="lines", line_color=ACCENT))
                        .update_layout(height=320, title="Rasio short volume seluruh universe (%)",
                                       margin=dict(l=10, r=10, t=40, b=10)), use_container_width=True)
        st.dataframe(dd.foreign_flow_table(c.repo), hide_index=True, use_container_width=True)
        tk = st.text_input("Detail saham (ticker)", "")
        if tk:
            f = dd.ticker_flows(c.repo, tk.strip().upper())
            if len(f):
                st.plotly_chart(go.Figure(go.Scatter(x=f["date"], y=f["short_ratio"] * 100, mode="lines+markers",
                                                     line_color=ACCENT))
                                .update_layout(height=300, title=f"{tk.upper()} rasio short volume FINRA (%)"),
                                use_container_width=True)
                st.dataframe(f, hide_index=True, use_container_width=True)

elif page == "Accumulation / Distribution":
    from app.research import dashboard_data as dd
    if research_ready():
        st.warning(research_ready())
        st.stop()
    st.title("Akumulasi / distribusi")
    st.caption("Bukti perilaku harga-volume (OBV, ADL, CMF, MFI, RVOL, CLV, VWAP, divergensi) + short flow FINRA bila ada. "
               "Bukan bukti identitas pembeli. STRONG hanya bila short flow tersedia & searah.")
    stages = st.multiselect("Tahap", ["EARLY_ACCUMULATION_WATCHLIST", "BREAKOUT_CONFIRMED", "DISTRIBUTION_WARNING", "NO_CLEAR_SIGNAL"],
                            ["EARLY_ACCUMULATION_WATCHLIST", "BREAKOUT_CONFIRMED", "DISTRIBUTION_WARNING"])
    tbl = dd.accumulation_table(c.repo, stages)
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    tk = st.selectbox("Bukti per emiten", [""] + tbl["ticker"].tolist() if len(tbl) else [""])
    if tk:
        st.dataframe(dd.accumulation_evidence(c.repo, tk), hide_index=True, use_container_width=True)

elif page == "Integrated Research":
    from app.research import dashboard_data as dd
    if research_ready():
        st.warning(research_ready())
        st.stop()
    from app.research.integrated_scoring import RANKINGS
    st.title("Integrated research")
    st.caption("Skor terpisah 0–100; composite = rata-rata berbobot skor yang TERSEDIA (bobot dinormalisasi ulang). "
               "Emiten dengan data kurang tidak diranking. VALUE dan SWING adalah keputusan terpisah.")
    t = dd.integrated_table(c.repo, c.cfg)
    if t.empty:
        st.info("Belum ada hasil daily.")
    else:
        name = st.selectbox("Ranking", RANKINGS, index=RANKINGS.index("integrated"))
        st.dataframe(dd.ranking_view(t, name, 30), hide_index=True, use_container_width=True)
        st.subheader("Keputusan VALUE vs SWING")
        st.dataframe(pd.crosstab(t["value_decision"], t["swing_decision"]), use_container_width=True)
        tk = st.selectbox("STOCK RESEARCH REPORT", [""] + sorted(t["ticker"].tolist()))
        if tk:
            from app.research.report import build_research, stock_report
            with st.spinner("Menghitung ulang (point-in-time) ..."):
                st.code(stock_report(build_research(c.cfg, c.repo, None, tickers=[tk]), tk), language=None)
