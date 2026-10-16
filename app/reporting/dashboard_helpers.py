"""Logika dashboard yang bisa diuji tanpa Streamlit (dashboard/app.py hanya merender).

Dibuat dari audit dashboard produksi 13 Okt 2026:
  - dashboard tersambung ke database lama tanpa tanda yang terlihat  → `database_banner`
  - KPI backtest tampil +0,0% / 0 trade untuk run evaluasi modul     → `backtest_kpis`, `module_variant_table`
  - BUY di luar batas rekomendasi regime tampil seperti rekomendasi  → `split_buys`
  - candlestick berlubang di hari libur bursa                        → `missing_business_days`
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit(root: Path = REPO_ROOT) -> str | None:
    """SHA commit yang ter-checkout (dibaca langsung dari .git, tanpa menjalankan git). None bila tidak tersedia."""
    try:
        head = (root / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:40] or None
        ref = head.split(" ", 1)[1].strip()
        f = root / ".git" / ref
        if f.exists():
            return f.read_text().strip()[:40] or None
        packed = root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0][:40]
    except Exception:
        return None
    return None


# Commit saat modul aplikasi (app/*) PERTAMA KALI dimuat di proses ini. Streamlit hanya menjalankan ulang
# dashboard/app.py ketika kode baru masuk; modul app/* yang sudah dimuat tetap versi lama sampai proses di-reboot.
LOADED_COMMIT = git_commit()


def stale_code_warning(current: str | None, loaded: str | None = None) -> str | None:
    loaded = LOADED_COMMIT if loaded is None else loaded
    if current and loaded and current != loaded:
        return (f"Versi kode baru terdeteksi ({current[:7]}), tetapi aplikasi masih memakai modul versi lama "
                f"({loaded[:7]}). Klik **Manage app → ⋮ → Reboot app** agar semua halaman memakai kode terbaru.")
    return None


# ----------------------------------------------------------------------------------------- database
def database_banner(db) -> tuple[str, str] | None:
    """(level, pesan) bila koneksi dashboard tidak sesuai keputusan proyek; None bila aman.

    level 'error' → dashboard berhenti (data yang tampil bukan data produksi)."""
    provider = getattr(db, "provider", getattr(db, "dialect", "?"))
    envs = {"supabase": "ALLOW_SUPABASE_DATABASE", "neon": "ALLOW_NEON_DATABASE"}
    if provider in envs and os.environ.get(envs[provider], "").strip().lower() not in ("1", "true", "yes"):
        return ("error", f"Dashboard tersambung ke database **{provider.capitalize()}** (milik proyek lain), bukan "
                         "database produksi versi US (Aiven). Ganti `DATABASE_URL` di Streamlit Cloud → app → "
                         "Settings → Secrets dengan service URI Aiven role `dashboard_ro`, lalu Reboot app.")
    if provider == "sqlite":
        return ("warning", "Dashboard memakai SQLite lokal (mode development).")
    return None


def connection_caption(db, latest_date) -> str:
    try:
        where = db.describe().split(":")[0]           # 'postgres[neon]' — tanpa host/kredensial
    except Exception:
        where = "?"
    return f"Database: {where} · data terakhir: {latest_date or '–'}"


# ----------------------------------------------------------------------------------------- backtest
_KPI_KEYS = ("total_return", "max_drawdown", "sharpe", "number_of_trades")


def backtest_kpis(metrics: dict | None) -> dict | None:
    """KPI ringkas untuk run walk-forward/live. None untuk format lain (mis. evaluasi modul) — JANGAN tampilkan 0."""
    if not isinstance(metrics, dict) or "total_return" not in metrics:
        return None
    return {k: metrics.get(k) for k in _KPI_KEYS}


def module_variant_table(metrics: dict | None) -> pd.DataFrame:
    """Run evaluasi modul (kind 'modules'): satu baris per periode × varian, metrik setelah biaya."""
    rows = []
    periods = (metrics or {}).get("periods") or {}
    for period, p in periods.items():
        for name, v in ((p or {}).get("variants") or {}).items():
            m = v.get("metrics") or {}
            rows.append({"periode": period, "varian": name, "status": v.get("status"),
                         "total_return": m.get("total_return"), "sharpe": m.get("sharpe"),
                         "max_drawdown": m.get("max_drawdown"), "trades": m.get("number_of_trades"),
                         "win_rate": m.get("win_rate"), "profit_factor": m.get("profit_factor"),
                         "data_coverage": v.get("data_coverage")})
    return pd.DataFrame(rows)


def default_backtest_run(runs: pd.DataFrame) -> int:
    """Indeks run yang dipilih default: walk-forward TEST terbaru (paling relevan), bukan run terakhir apa pun."""
    if runs is None or runs.empty:
        return 0
    ids = runs["run_id"].astype(str).tolist()
    for prefix in ("wf-test", "wf-validation", "live"):
        for i, r in enumerate(ids):
            if r.startswith(prefix):
                return i
    return 0


def strategy_verdict(metrics: dict | None, benchmark: dict | None) -> tuple[str, str] | None:
    """Kalimat jujur tentang kinerja strategi pada run TEST terbaru (setelah biaya). None bila tidak ada data."""
    k = backtest_kpis(metrics)
    if not k or k["total_return"] is None:
        return None
    ihsg = ((benchmark or {}).get("ihsg_buy_hold") or {}).get("total_return")
    tech = ((benchmark or {}).get("technical_only") or {}).get("total_return")
    period = f"{metrics.get('start', '?')} s/d {metrics.get('end', '?')}"
    parts = [f"Backtest out-of-sample {period}, setelah biaya: strategi {k['total_return']:+.1%}"
             f" ({k['number_of_trades']} trade, Sharpe {k['sharpe']:.2f})"]
    if tech is not None:
        parts.append(f"teknikal-saja {tech:+.1%}")
    if ihsg is not None:
        parts.append(f"S&P 500 {ihsg:+.1%}")
    msg = ", ".join(parts) + "."
    if k["total_return"] < 0:
        return ("warning", msg + " Strategi masih RUGI secara absolut pada periode ini — perlakukan sinyal sebagai "
                                 "bahan riset, bukan instruksi transaksi.")
    return ("info", msg + " Satu periode belum membuktikan strategi layak.")


# ----------------------------------------------------------------------------------------- sinyal
def split_buys(buy_predictions: pd.DataFrame, ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(BUY yang masuk ranking, BUY yang tertahan batas rekomendasi regime). Urutan ranking dipertahankan."""
    if buy_predictions is None or buy_predictions.empty:
        e = pd.DataFrame(columns=list(getattr(buy_predictions, "columns", [])))
        return e, e
    ranked_tickers = list(ranked["ticker"]) if ranked is not None and len(ranked) else []
    inside = buy_predictions[buy_predictions["ticker"].isin(ranked_tickers)].copy()
    inside["_o"] = inside["ticker"].map({t: i for i, t in enumerate(ranked_tickers)})
    inside = inside.sort_values("_o").drop(columns="_o")
    outside = buy_predictions[~buy_predictions["ticker"].isin(ranked_tickers)]
    return inside.reset_index(drop=True), outside.reset_index(drop=True)


# ----------------------------------------------------------------------------------------- grafik
def missing_business_days(dates) -> list[str]:
    """Hari kerja (Sen–Jum) tanpa data di rentang grafik → rangebreaks plotly (libur bursa, suspensi)."""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if d.empty:
        return []
    have = set(d)
    return [x.strftime("%Y-%m-%d") for x in pd.bdate_range(d.min(), d.max()) if x not in have]
