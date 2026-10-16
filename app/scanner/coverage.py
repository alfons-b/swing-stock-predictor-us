"""Cakupan pemindaian seluruh universe (full-market coverage).

Setiap emiten AKTIF di universe mendapat tepat satu status per tanggal scan, beserta alasannya:

  SUCCESS            data lengkap, histori cukup, lolos aturan universe → boleh mendapat sinyal
  PARTIAL_DATA       ada baris hari scan tetapi fitur inti kosong / baris ditolak cleaner
  INSUFFICIENT_DATA  histori terlalu pendek untuk indikator & model (mis. IPO baru) atau belum ada data sama sekali
  PROVIDER_ERROR     provider gagal secara teknis untuk emiten ini pada run ini (bukan "tidak ada transaksi")
  EXCLUDED           data valid, tetapi dikecualikan aturan universe: suspensi, tidak likuid, harga < minimum,
                     board yang eksekusinya tidak dimodelkan (OTC / pink sheet)
  STALE_DATA         data terakhir lebih tua dari tanggal scan (suspensi panjang, provider tidak memperbarui)

Prinsip: emiten yang statusnya bukan SUCCESS TIDAK boleh menjadi BUY (lihat `apply_coverage_gate`). Skor teknikal
mereka tetap disimpan untuk transparansi, tetapi tidak pernah diperlakukan sebagai sinyal valid.

Coverage dihitung dari frame fitur hari scan (sebelum ranking), sehingga gerbang berlaku sebelum peringkat dibuat.
"""
from __future__ import annotations

import pandas as pd

from app.config import get

STATUSES = ["SUCCESS", "PARTIAL_DATA", "INSUFFICIENT_DATA", "PROVIDER_ERROR", "EXCLUDED", "STALE_DATA"]
# Fitur yang wajib ada agar setup, stop-loss, sizing, dan model bermakna.
CORE_FEATURES = ["close", "atr_pct", "ema20", "ema50", "rsi14", "adx", "volume_ratio", "avg_value20"]
DEFAULT_EXCLUDED_BOARDS = ["OTC", "Pink", "Grey"]


def _norm_board(b) -> str:
    return str(b or "").strip().lower()


def build_coverage(universe: pd.DataFrame, latest: pd.DataFrame, day: pd.DataFrame | None, scan_date,
                   cfg: dict, provider_failed: dict | None = None, provider_no_data: list | None = None) -> pd.DataFrame:
    """
    universe : stocks (ticker, is_active, board, ...) — hanya yang aktif yang dinilai
    latest   : Repository.latest_price_dates() → ticker, last_date, n_rows
    day      : frame fitur PADA tanggal scan (satu baris per emiten yang punya data hari itu)
    """
    scan_date = pd.Timestamp(scan_date).normalize()
    provider_failed = provider_failed or {}
    no_data = set(provider_no_data or [])
    min_hist = int(get(cfg, "quality.min_history_days", 250))
    min_price = float(get(cfg, "quality.min_price", 50))
    excl_boards = {_norm_board(b) for b in (get(cfg, "coverage.excluded_boards", DEFAULT_EXCLUDED_BOARDS) or [])}

    act = universe[universe["is_active"].astype(bool)].copy()
    lt = latest.set_index("ticker") if len(latest) else pd.DataFrame(columns=["last_date", "n_rows"])
    dd = day.set_index("ticker") if day is not None and len(day) else pd.DataFrame()
    core = [c for c in CORE_FEATURES if c in dd.columns]

    rows = []
    for r in act.itertuples():
        t = r.ticker
        last = lt["last_date"].get(t) if t in lt.index else pd.NaT
        n_rows = int(lt["n_rows"].get(t) or 0) if t in lt.index else 0
        last = pd.Timestamp(last) if pd.notna(last) else pd.NaT
        board = getattr(r, "board", None)
        rec = {"ticker": t, "last_data_date": None if pd.isna(last) else last.strftime("%Y-%m-%d"),
               "history_rows": n_rows, "board": board, "analyzed": t in dd.index}

        if t not in dd.index:
            if t in provider_failed:
                status, reason = "PROVIDER_ERROR", f"provider: {provider_failed[t]}"
            elif n_rows == 0 or pd.isna(last):
                status, reason = "INSUFFICIENT_DATA", "belum ada histori harga di database"
            elif last < scan_date:
                status = "STALE_DATA"
                reason = f"data terakhir {last.date()}"
                if t in no_data:
                    reason += " — provider tidak mengembalikan transaksi baru (suspensi/tidak diperdagangkan?)"
            else:
                status, reason = "PARTIAL_DATA", "baris harga hari scan ditolak validasi/cleaner (OHLC tidak valid / di luar periode listing)"
            rows.append({**rec, "status": status, "reason": reason})
            continue

        x = dd.loc[t]
        if isinstance(x, pd.DataFrame):          # duplikat (seharusnya tidak terjadi) → ambil terakhir
            x = x.iloc[-1]
        days_listed = x.get("days_listed")
        missing = [c for c in core if pd.isna(x.get(c))]
        excl = []
        if _norm_board(board) in excl_boards:
            excl.append(f"board {board} (OTC/non-bursa — eksekusi tidak dimodelkan)")
        if bool(x.get("is_suspended", False)):
            excl.append("suspensi / tidak ada transaksi")
        if "is_liquid" in x and not bool(x.get("is_liquid")):
            excl.append("tidak likuid (di bawah batas nilai/volume transaksi)")
        if pd.notna(x.get("close")) and float(x["close"]) < min_price:
            excl.append(f"harga < {min_price:.0f}")

        if pd.notna(days_listed) and int(days_listed) < min_hist:
            status, reason = "INSUFFICIENT_DATA", f"histori {int(days_listed)} hari bursa < minimum {min_hist}"
        elif missing:
            status, reason = "PARTIAL_DATA", "fitur inti kosong: " + ", ".join(missing)
        elif excl:
            status, reason = "EXCLUDED", "; ".join(excl)
        else:
            status, reason = "SUCCESS", ""
        rows.append({**rec, "status": status, "reason": reason})

    out = pd.DataFrame(rows, columns=["ticker", "status", "reason", "last_data_date", "history_rows", "board", "analyzed"])
    out.insert(0, "scan_date", scan_date.strftime("%Y-%m-%d"))
    return out


def coverage_summary(cov: pd.DataFrame) -> dict:
    total = int(len(cov))
    counts = {s: int((cov["status"] == s).sum()) for s in STATUSES}
    analyzed = int(cov["analyzed"].astype(bool).sum()) if total else 0
    top = (cov[cov["status"] != "SUCCESS"].assign(r=cov["reason"].str.split(" —|;|:").str[0].str.strip())
           .groupby(["status", "r"]).size().sort_values(ascending=False).head(8))
    return {
        "scan_date": cov["scan_date"].iloc[0] if total else None,
        "total_universe": total,
        "processed": analyzed,                 # emiten yang punya data hari scan dan dianalisis
        "failed": counts["PROVIDER_ERROR"],
        "incomplete": counts["PARTIAL_DATA"] + counts["INSUFFICIENT_DATA"] + counts["STALE_DATA"],
        "excluded": counts["EXCLUDED"],
        "signal_eligible": counts["SUCCESS"],
        "coverage_pct": round(100.0 * analyzed / total, 1) if total else 0.0,
        "eligible_pct": round(100.0 * counts["SUCCESS"] / total, 1) if total else 0.0,
        "status_counts": counts,
        "top_reasons": [{"status": s, "reason": r, "n": int(n)} for (s, r), n in top.items()],
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }


def apply_coverage_gate(sig: pd.DataFrame, cov: pd.DataFrame | None) -> pd.DataFrame:
    """BUY hanya untuk status SUCCESS. Selain itu diturunkan ke WAIT dengan alasan eksplisit (tidak dihapus)."""
    if cov is None or cov.empty or sig.empty:
        return sig
    st = dict(zip(cov["ticker"], cov["status"]))
    status = sig["ticker"].map(st).fillna("NOT_IN_UNIVERSE")
    bad = (sig["decision"] == "BUY") & (status != "SUCCESS")
    if not bad.any():
        return sig
    sig = sig.copy()
    sig.loc[bad, "decision"] = "WAIT"
    sig["reject_reasons"] = [list(r) + [f"coverage_{s.lower()}"] if b else r
                             for r, b, s in zip(sig["reject_reasons"], bad, status)]
    return sig


def coverage_csv(cov: pd.DataFrame) -> bytes:
    order = {s: i for i, s in enumerate(STATUSES)}
    return (cov.assign(_o=cov["status"].map(order)).sort_values(["_o", "ticker"]).drop(columns="_o")
            .to_csv(index=False).encode("utf-8"))


def coverage_markdown(summary: dict) -> list[str]:
    if not summary:
        return []
    c = summary["status_counts"]
    L = ["## Cakupan pemindaian (seluruh universe)", "",
         f"Universe aktif **{summary['total_universe']}** · dianalisis **{summary['processed']}** "
         f"({summary['coverage_pct']}%) · layak sinyal **{summary['signal_eligible']}** ({summary['eligible_pct']}%)", "",
         "| Status | Jumlah |", "|---|---|"] + [f"| {s} | {c[s]} |" for s in STATUSES]
    if summary.get("top_reasons"):
        L += ["", "Alasan terbanyak (bukan SUCCESS):", ""] + \
             [f"- {x['status']}: {x['reason']} ({x['n']})" for x in summary["top_reasons"]]
    return L + [""]
