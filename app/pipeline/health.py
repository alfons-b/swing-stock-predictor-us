"""Health check (§53) — dipakai CLI `health`, awal pipeline daily, dan dashboard (halaman Data Health)."""
from __future__ import annotations

import json
import sys

import pandas as pd

from app import FEATURE_VERSION
from app.config import get, is_production
from app.pipeline.freshness import data_freshness

REQUIRED_PACKAGES = ["numpy", "pandas", "sklearn", "scipy", "yaml", "joblib"]


def check_environment(cfg: dict) -> list[str]:
    problems = []
    if sys.version_info < (3, 10):
        problems.append(f"Python >= 3.10 dibutuhkan (sekarang {sys.version.split()[0]})")
    import importlib
    for p in REQUIRED_PACKAGES:
        try:
            importlib.import_module(p)
        except ImportError:
            problems.append(f"Paket {p} belum terpasang")
    url = str(get(cfg, "database.url", ""))
    if is_production(cfg) and not url.startswith(("postgres://", "postgresql://")):
        problems.append("environment=production tetapi DATABASE_URL bukan PostgreSQL (set GitHub Secret DATABASE_URL)")
    if url.startswith(("postgres://", "postgresql://")):
        try:
            importlib.import_module("psycopg")
        except ImportError:
            problems.append("psycopg belum terpasang (pip install 'psycopg[binary]')")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(get(cfg, "calendar.timezone", "America/New_York"))
    except Exception:
        problems.append("Timezone America/New_York tidak tersedia (Windows: pip install tzdata)")
    return problems


def health_check(ctx, check_provider: bool = True) -> dict:
    cfg, repo = ctx.cfg, ctx.repo
    out: dict = {"checks": {}}

    def put(name, status, detail=""):
        out["checks"][name] = {"status": status, "detail": detail}

    env = check_environment(cfg)
    put("ENVIRONMENT", "OK" if not env else "FAIL", "; ".join(env))
    try:
        ctx.db.scalar("SELECT 1")
        put("DATABASE", "OK", ctx.db.describe())
        _provider_check(cfg, ctx.db, put)
        from app.database.maintenance import size_status
        sz = size_status(cfg, ctx.db)
        out["database_size"] = sz
        if sz["read_only"]:
            put("DATABASE SIZE", "FAIL", f"{sz['bytes'] / 1e6:.0f} MB — database READ-ONLY (kuota terlampaui): "
                                         "jalankan workflow Database Maintenance dengan opsi full")
        else:
            put("DATABASE SIZE", sz["status"], f"{sz['bytes'] / 1e6:.0f} MB dari {sz['limit_bytes'] / 1e6:.0f} MB "
                                               f"({sz['ratio']:.0%})" + (" — jalankan Database Maintenance" if sz["status"] != "OK" else ""))
    except Exception as e:
        put("DATABASE", "FAIL", f"{type(e).__name__}: {e}"[:200])
        out["ok"] = False
        return out

    provider_latest = None
    if check_provider:
        try:
            provider_latest, src = ctx.chain.latest_available_date()
            put("MARKET DATA PROVIDER", "OK" if provider_latest is not None else "FAIL",
                f"{src}: data terakhir {provider_latest.date()}" if provider_latest is not None else "tidak ada provider yang merespons")
        except Exception as e:
            put("MARKET DATA PROVIDER", "FAIL", f"{type(e).__name__}: {e}"[:200])
    n_active = int(ctx.db.scalar("SELECT COUNT(*) FROM stocks WHERE is_active = ?", (True,)) or 0)
    from app.data.universe_file import resolve_universe_file
    try:
        uf, is_example = resolve_universe_file(cfg)
    except Exception:
        uf, is_example = None, False
    primary_csv = bool(get(cfg, "market_data.providers")) and \
        sorted(get(cfg, "market_data.providers"), key=lambda p: int(p.get("priority", 50)))[0].get("type") == "csv"
    if n_active == 0:
        put("UNIVERSE", "EMPTY", "0 emiten aktif")
    elif is_example and not primary_csv:
        put("UNIVERSE", "WARN", f"{n_active} saham aktif — DAFTAR CONTOH ({uf.name}), bukan seluruh universe: "
                                "pastikan Wikipedia/GitHub bisa diakses atau simpan daftar Anda sebagai config/universe.csv")
    else:
        index_src = "us_index" in (get(cfg, "universe.sources") or []) and not primary_csv
        src = (f" (sumber: konstituen indeks {', '.join(get(cfg, 'universe.indexes', []) or [])} via Wikipedia; cadangan "
               f"{uf.name if uf else '-'})") if index_src else (f" (sumber: {uf.name})" if uf and not primary_csv else "")
        put("UNIVERSE", "OK", f"{n_active} saham aktif" + src)

    fr = data_freshness(cfg, repo, ctx.calendar, provider_latest)
    out["freshness"] = fr
    put("LATEST MARKET DATE", "INFO", fr["latest_expected_market_date"])
    put("DATABASE LATEST DATE", "INFO", str(fr["latest_database_date"]))
    put("DATA FRESHNESS", {"UP_TO_DATE": "OK", "UPDATE_REQUIRED": "WARN"}.get(fr["status"], "FAIL"), fr["status"])

    try:
        active = ctx.models.active_version()
    except Exception as e:
        active = None
        put("ACTIVE MODEL", "FAIL", str(e)[:200])
    if active:
        put("ACTIVE MODEL", "OK", active["version"])
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(active["promoted_at"] or active["created_at"])).days
        status, detail = "OK", f"umur {age} hari, train s/d {active['train_end']}"
        if active.get("feature_version") != FEATURE_VERSION:
            status, detail = "INCOMPATIBLE", f"feature_version {active.get('feature_version')} != {FEATURE_VERSION} — retrain"
        elif age > 2 * int(get(cfg, "model.retrain_frequency_days", 30)):
            status = "STALE"
        elif "BASELINE" in str(ctx.db.scalar("SELECT notes FROM model_versions WHERE version = ?", (active["version"],)) or ""):
            status, detail = "WARN", detail + " (baseline: gerbang promosi belum lolos)"
        put("MODEL STATUS", status, detail)
        out["active_model"] = active["version"]
    else:
        put("ACTIVE MODEL", "MISSING", "jalankan `python main.py setup` / workflow Setup")
        put("MODEL STATUS", "MISSING", "")
    _research_checks(cfg, ctx.db, put)
    try:
        from app.pipeline.evaluate import live_calibration
        cal = live_calibration(repo)
        out["live_calibration"] = cal
        limit = 1.5 * float(get(cfg, "promotion.max_ece", 0.06))
        if cal["status"] != "OK":
            put("LIVE CALIBRATION", "INFO", f"{cal['n']} prediksi terevaluasi (120 hari) — belum cukup untuk dinilai")
        else:
            put("LIVE CALIBRATION", "OK" if cal["ece_bullish"] <= limit else "WARN",
                f"ECE {cal['ece_bullish']:.3f} (batas {limit:.3f}), prob rata-rata {cal['mean_prob_bullish']:.1%} vs "
                f"realisasi {cal['realized_bullish_rate']:.1%}, n={cal['n']}"
                + ("" if cal["ece_bullish"] <= limit else " — model tidak terkalibrasi di data live: pertimbangkan retrain"))
    except Exception as e:
        put("LIVE CALIBRATION", "INFO", f"{type(e).__name__}: {e}"[:200])
    last = repo.last_run("daily")
    put("LAST PIPELINE", last["status"] if last else "NONE",
        f"{last['run_id']} @ {last['started_at']}" if last else "belum pernah dijalankan")
    critical = ["ENVIRONMENT", "DATABASE"]
    out["ok"] = all(out["checks"][c]["status"] == "OK" for c in critical if c in out["checks"])
    return out


STRICT_CHECKS = ["ENVIRONMENT", "DATABASE", "DATABASE PROVIDER", "DATABASE SIZE", "DATABASE WRITE"]


def write_probe(db) -> tuple[str, str]:
    """Tulis lalu hapus satu baris log. Mendeteksi penyimpanan yang diblokir (kuota Neon/Aiven habis, Supabase
    read-only, role tanpa izin tulis) — hal yang tidak terlihat dari SELECT 1."""
    marker = "health-write-probe"
    try:
        db.execute("INSERT INTO system_logs (run_id, ts, level, logger, message) VALUES (?, ?, ?, ?, ?)",
                   (marker, pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "INFO", "health", marker))
        db.execute("DELETE FROM system_logs WHERE run_id = ?", (marker,))
        return "OK", "insert + delete berhasil"
    except Exception as e:
        return "FAIL", f"penulisan gagal: {type(e).__name__}: {str(e)[:160]}"



def strict_failures(h: dict) -> list[str]:
    """Pemeriksaan untuk workflow Database Health Check: gagal juga saat penyimpanan hampir/sudah diblokir."""
    return [k for k in STRICT_CHECKS if h["checks"].get(k, {}).get("status") == "FAIL"]


def _provider_check(cfg, db, put) -> None:
    """Provider database production. Keputusan proyek (docs/DATABASE_DECISION.md): Aiven Free, database terpisah;
    Neon (versi BEI) dan Supabase (aplikasi lain) dipakai proyek lain pemilik repo."""
    if db.dialect != "postgres":
        if is_production(cfg):
            put("DATABASE PROVIDER", "FAIL", "production wajib PostgreSQL eksternal")
        return
    prof = db.profile
    if prof.name in ("supabase", "neon"):
        put("DATABASE PROVIDER", "WARN", f"{prof.name.capitalize()} terdeteksi — database ini milik proyek lain; versi US "
                                         "harus memakai Aiven (docs/SETUP_GUIDE.md Langkah 2)")
    elif prof.name == "aiven":
        put("DATABASE PROVIDER", "OK", f"{prof.name} (paket gratis: batas efektif {prof.free_size_limit_mb:.0f} MB; "
                                       f"bila terlampaui: {prof.on_limit})")
    else:
        put("DATABASE PROVIDER", "INFO", f"{prof.name} — batas ukuran dari DB_SIZE_LIMIT_MB / default "
                                         f"{prof.free_size_limit_mb:.0f} MB")


def _research_checks(cfg, db, put) -> None:
    """Status data riset — informasi (WARN), tidak pernah menggagalkan pipeline swing."""
    try:
        n_act = int(db.scalar("SELECT COUNT(*) FROM stocks WHERE is_active = ?", (True,)) or 0)
        n_f = int(db.scalar("SELECT COUNT(DISTINCT f.stock_id) FROM financial_statements f JOIN stocks s ON s.id = f.stock_id "
                            "WHERE s.is_active = ?", (True,)) or 0)
        latest_pe = db.scalar("SELECT MAX(period_end) FROM financial_statements")
        stale = int(get(cfg, "fundamentals.stale_after_days", 450))
        if n_f == 0:
            put("FUNDAMENTALS", "WARN", "belum ada laporan keuangan → valuasi INSUFFICIENT_DATA (lihat data_sources)")
        else:
            age = (pd.Timestamp.now() - pd.Timestamp(latest_pe)).days if latest_pe else None
            put("FUNDAMENTALS", "OK" if n_f >= 0.5 * max(n_act, 1) and (age or 0) <= stale else "WARN",
                f"{n_f}/{n_act} emiten punya laporan; periode terbaru {str(latest_pe)[:10]}")
        asof = get(cfg, "valuation.assumptions_as_of")
        try:
            from app.data.alpha_vantage_update import risk_free_from_treasury
            from types import SimpleNamespace
            rf, rf_src = risk_free_from_treasury(cfg, SimpleNamespace(db=db))
        except Exception:
            rf, rf_src = None, None
        if rf is not None:
            put("VALUATION ASSUMPTIONS", "OK", f"risk-free {rf} otomatis dari {rf_src}; ERP "
                f"{get(cfg, 'valuation.equity_risk_premium')} (ditinjau {asof})")
            asof = None
        if asof:
            age = (pd.Timestamp.now() - pd.Timestamp(str(asof))).days
            put("VALUATION ASSUMPTIONS", "OK" if age <= 180 else "WARN",
                f"risk-free {get(cfg, 'valuation.risk_free_rate')}, ERP {get(cfg, 'valuation.equity_risk_premium')} "
                f"ditinjau {asof}" + ("" if age <= 180 else " — > 180 hari: perbarui config/research.yaml"))
        _alpha_vantage_health(cfg, db, put)
        last_flow = db.scalar("SELECT MAX(date) FROM foreign_flow_history")
        if not last_flow:
            put("SHORT FLOW", "WARN", "FOREIGN_FLOW_UNAVAILABLE — data FINRA belum ada; skor/ranking flow tidak dihitung")
        else:
            lag = (pd.Timestamp.now() - pd.Timestamp(last_flow)).days
            put("SHORT FLOW", "OK" if lag <= 7 else "WARN", f"data terakhir {str(last_flow)[:10]}"
                + ("" if lag <= 7 else " — basi, tidak dipakai dalam skor"))
    except Exception as e:  # tabel riset belum dimigrasi dsb.
        put("RESEARCH DATA", "WARN", f"{type(e).__name__}: {e}"[:200])


def format_health(h: dict) -> str:
    lines = []
    for name, c in h["checks"].items():
        d = f"  ({c['detail']})" if c["detail"] and c["status"] not in ("INFO",) else ""
        val = c["detail"] if c["status"] == "INFO" else c["status"]
        lines.append(f"{name + ':':<24}{val}{d}")
    return "\n".join(lines)


def health_json(h: dict) -> str:
    return json.dumps(h, default=str, indent=2)


def _alpha_vantage_health(cfg: dict, db, put) -> None:
    """Status data Alpha Vantage: key, kuota 24 jam, cakupan surprise EPS, kalender, yield, berita."""
    import os
    key = (get(cfg, "alpha_vantage.api_key") or os.environ.get("ALPHAVANTAGE_API_KEY") or "").strip()
    try:
        since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)).isoformat()
        used = int(db.scalar("SELECT COUNT(*) FROM api_usage WHERE provider = 'alpha_vantage' AND called_at > ?",
                             (since,)) or 0)
        cov = int(db.scalar("SELECT COUNT(DISTINCT stock_id) FROM earnings_surprises") or 0)
        n_act = int(db.scalar("SELECT COUNT(*) FROM stocks WHERE is_active = ?", (True,)) or 0)
        cal = db.scalar("SELECT MAX(as_of) FROM earnings_calendar")
        ust = db.scalar("SELECT MAX(date) FROM macro_series WHERE series = 'UST10YEAR'")
        news = db.scalar("SELECT MAX(published_at) FROM news_items")
    except Exception as e:
        put("ALPHA VANTAGE", "INFO", f"tabel belum tersedia ({type(e).__name__})")
        return
    if not key and not cov:
        put("ALPHA VANTAGE", "INFO", "tidak aktif — set GitHub Secret ALPHAVANTAGE_API_KEY (opsional, gratis)")
        return
    limit = int(get(cfg, "alpha_vantage.daily_limit", 25) or 25)
    cal_age = (pd.Timestamp.now() - pd.Timestamp(str(cal)[:10])).days if cal else None
    status = "OK" if cal_age is not None and cal_age <= 4 else "WARN"
    put("ALPHA VANTAGE", status,
        f"kuota 24 jam {used}/{limit}; surprise EPS {cov}/{n_act} saham; kalender {str(cal)[:10] if cal else '-'}"
        f"; Treasury 10Y {str(ust)[:10] if ust else '-'}; berita {str(news)[:10] if news else '-'}"
        + ("" if status == "OK" else " — kalender rilis basi/kosong: cek langkah 'Alpha Vantage' di workflow Daily"))
