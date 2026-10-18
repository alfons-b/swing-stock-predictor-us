"""Alpha Vantage (paket gratis) — data yang TIDAK bisa didapat gratis dari SEC/Yahoo/FINRA.

Kuota gratis: 25 request per hari, 5 per menit. Dengan ±900 saham, setiap request harus "membeli" sinyal yang
paling bernilai. Alokasi (config/sources.yaml → alpha_vantage):

  Fungsi              Request    Dipakai untuk
  EARNINGS_CALENDAR   1/hari     Tanggal rilis laba BERIKUTNYA yang pasti untuk SEMUA saham (1 CSV) → filter
                                 blackout earnings tidak lagi memakai perkiraan; konsensus EPS disimpan per hari
                                 (riwayat revisi analis terkumpul mulai sekarang).
  TREASURY_YIELD      ±1/hari    Imbal hasil 10Y/2Y/3M → risk-free valuasi otomatis + kurva yield (regime makro).
  NEWS_SENTIMENT      0          PREMIUM sejak 2026 (terverifikasi 18 Okt) — nonaktif; aktifkan via
                                 news_calls_per_run bila paket berbayar. Fungsi premium otomatis dilewati 30 hari.
  EARNINGS            sisanya    EPS aktual vs estimasi analis per kuartal (riwayat penuh dalam 1 request) →
                                 standardized unexpected earnings (SUE), sinyal PEAD klasik (Bernard & Thomas 1989).

Prinsip:
- Kuota dihitung dari tabel api_usage dalam jendela bergulir 24 jam (aman apa pun jam reset Alpha Vantage), lintas
  workflow. Respons "rate limit" menghentikan semua panggilan sisa run.
- API key HANYA dari env ALPHAVANTAGE_API_KEY (GitHub Secret). Pesan error disaring agar key tidak pernah tercetak.
- Semua data disimpan point-in-time (tanggal tersedia publik), sehingga bisa dipakai jujur di backtest.
"""
from __future__ import annotations

import os
import re
import time
from io import StringIO

import numpy as np
import pandas as pd

from app.config import get
from app.data.tickers import normalize_ticker
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
PROVIDER = "alpha_vantage"


class AlphaVantageError(RuntimeError):
    pass


class QuotaExhausted(AlphaVantageError):
    pass


class PremiumOnly(AlphaVantageError):
    pass


def _redact(text: str, key: str | None) -> str:
    s = str(text)
    if key:
        s = s.replace(key, "***")
    return re.sub(r"apikey=[^&\s]+", "apikey=***", s)


def _num(v) -> float:
    if v is None:
        return np.nan
    s = str(v).strip()
    if s in ("", "None", "none", "null", ".", "-", "N/A"):
        return np.nan
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return np.nan


def _ticker(v) -> str | None:
    try:
        return normalize_ticker(str(v).replace(".", "-"))
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------------------------------------ parser (murni)
def parse_earnings(js: dict, ticker: str) -> pd.DataFrame:
    """EARNINGS → satu baris per kuartal yang sudah dilaporkan."""
    rows = []
    for q in (js or {}).get("quarterlyEarnings") or []:
        rd = pd.to_datetime(q.get("reportedDate"), errors="coerce")
        if pd.isna(rd):
            continue
        rows.append({"ticker": ticker, "fiscal_date_ending": pd.to_datetime(q.get("fiscalDateEnding"), errors="coerce"),
                     "reported_date": rd, "report_time": (q.get("reportTime") or "").strip().lower() or None,
                     "reported_eps": _num(q.get("reportedEPS")), "estimated_eps": _num(q.get("estimatedEPS")),
                     "surprise": _num(q.get("surprise")), "surprise_pct": _num(q.get("surprisePercentage"))})
    cols = ["ticker", "fiscal_date_ending", "reported_date", "report_time", "reported_eps", "estimated_eps",
            "surprise", "surprise_pct"]
    df = pd.DataFrame(rows, columns=cols)
    # surprise kosong tetapi EPS & estimasi ada → hitung sendiri
    miss = df["surprise"].isna() & df["reported_eps"].notna() & df["estimated_eps"].notna()
    df.loc[miss, "surprise"] = df.loc[miss, "reported_eps"] - df.loc[miss, "estimated_eps"]
    return df.dropna(subset=["fiscal_date_ending"]).drop_duplicates(["ticker", "fiscal_date_ending"], keep="first")


def parse_calendar(text: str) -> pd.DataFrame:
    """EARNINGS_CALENDAR (CSV) → ticker, report_date, fiscal_date_ending, estimate, currency, time_of_day."""
    cols = ["ticker", "report_date", "fiscal_date_ending", "estimate", "currency", "time_of_day"]
    if not text or not text.strip() or text.lstrip().startswith("{"):
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(StringIO(text), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    low = {c.lower(): c for c in raw.columns}
    if "symbol" not in low or "reportdate" not in low:
        raise AlphaVantageError(f"format EARNINGS_CALENDAR tidak dikenal: kolom {list(raw.columns)[:8]}")
    out = pd.DataFrame({
        "ticker": raw[low["symbol"]].map(_ticker),
        "report_date": pd.to_datetime(raw[low["reportdate"]], errors="coerce"),
        "fiscal_date_ending": pd.to_datetime(raw[low["fiscaldateending"]], errors="coerce") if "fiscaldateending" in low
        else pd.NaT,
        "estimate": raw[low["estimate"]].map(_num) if "estimate" in low else np.nan,
        "currency": raw[low["currency"]] if "currency" in low else None,
        "time_of_day": raw[low["timeoftheday"]].str.strip().str.lower() if "timeoftheday" in low else None,
    })
    return out.dropna(subset=["ticker", "report_date"])[cols]


def parse_series(js: dict, series: str) -> pd.DataFrame:
    """TREASURY_YIELD / FEDERAL_FUNDS_RATE → series, date, value (persen)."""
    data = (js or {}).get("data") or []
    df = pd.DataFrame({"series": series, "date": pd.to_datetime([d.get("date") for d in data], errors="coerce"),
                       "value": [_num(d.get("value")) for d in data]})
    return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)


def parse_news(js: dict, min_relevance: float = 0.25) -> pd.DataFrame:
    """NEWS_SENTIMENT → satu baris per (artikel, ticker) yang relevan. time_published = waktu AS bagian timur."""
    rows = []
    for a in (js or {}).get("feed") or []:
        ts = pd.to_datetime(a.get("time_published"), format="%Y%m%dT%H%M%S", errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(a.get("time_published"), errors="coerce")
        if pd.isna(ts):
            continue
        for t in a.get("ticker_sentiment") or []:
            rel = _num(t.get("relevance_score"))
            if np.isnan(rel) or rel < min_relevance:
                continue
            tk = _ticker(t.get("ticker"))
            if not tk or ":" in str(t.get("ticker")):          # CRYPTO:BTC, FOREX:USD dll. bukan saham
                continue
            rows.append({"ticker": tk, "published_at": ts, "headline": (a.get("title") or "")[:300],
                         "url": a.get("url") or "", "source": a.get("source") or "",
                         "sentiment": float(np.clip(_num(t.get("ticker_sentiment_score")), -1, 1)),
                         "relevance": rel})
    return pd.DataFrame(rows, columns=["ticker", "published_at", "headline", "url", "source", "sentiment", "relevance"])


# ------------------------------------------------------------------------------------------------ klien + kuota
class AlphaVantageClient:
    def __init__(self, cfg: dict, repo=None, http_get=None, sleep=time.sleep):
        self.cfg = cfg
        self.repo = repo
        self.key = (get(cfg, "alpha_vantage.api_key") or os.environ.get("ALPHAVANTAGE_API_KEY") or "").strip()
        if self.key.startswith("${"):
            self.key = ""
        self.daily_limit = int(get(cfg, "alpha_vantage.daily_limit", 25) or 25)
        self.min_interval = 60.0 / max(1, int(get(cfg, "alpha_vantage.per_minute_limit", 5) or 5)) + 0.6
        self._http_get = http_get
        self._sleep = sleep
        self._last = 0.0
        self._exhausted = False
        self.calls = 0

    def configured(self) -> bool:
        return bool(self.key) and str(get(self.cfg, "alpha_vantage.enabled", True)).lower() in ("true", "1", "yes")

    def used_24h(self) -> int:
        if self.repo is None:
            return self.calls
        since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)).isoformat()
        try:
            return int(self.repo.db.scalar("SELECT COUNT(*) FROM api_usage WHERE provider = ? AND called_at > ?",
                                           (PROVIDER, since)) or 0)
        except Exception:
            return self.calls

    def remaining(self) -> int:
        return 0 if self._exhausted else max(0, self.daily_limit - self.used_24h())

    def _record(self, function: str, symbol: str | None, status: str) -> None:
        self.calls += 1
        if self.repo is None:
            return
        try:
            self.repo.db.execute("INSERT INTO api_usage (provider, called_at, function, symbol, status) VALUES (?, ?, ?, ?, ?)",
                                 (PROVIDER, pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), function, symbol,
                                  status[:60]))
        except Exception as e:  # pencatatan tidak boleh menggagalkan pipeline
            log.debug("catat api_usage gagal: %s", e)

    def _get(self, params: dict):
        if self._http_get is not None:
            return self._http_get(BASE_URL, params)
        import requests
        try:
            r = requests.get(BASE_URL, params=params, timeout=40, headers={"User-Agent": "swing-stock-predictor-us/1.2"})
        except Exception as e:
            raise AlphaVantageError(_redact(f"{type(e).__name__}: {e}", self.key)) from None
        if r.status_code != 200:
            raise AlphaVantageError(f"HTTP {r.status_code}")
        return r.text

    def call(self, function: str, symbol: str | None = None, **params):
        """Satu request. Mengembalikan dict (JSON) atau str (CSV)."""
        if not self.configured():
            raise AlphaVantageError("ALPHAVANTAGE_API_KEY belum diset")
        if self.remaining() <= 0:
            raise QuotaExhausted("kuota 24 jam Alpha Vantage habis")
        wait = self._last + self.min_interval - time.monotonic()
        if self._last and wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
        q = {"function": function, "apikey": self.key, **params}
        if symbol:
            q["symbol"] = symbol
        try:
            text = self._get(q)
        except AlphaVantageError as e:
            self._record(function, symbol, f"ERROR {e}")
            raise
        body = text.lstrip() if isinstance(text, str) else text
        js = None
        if isinstance(body, str) and body.startswith("{"):
            import json
            js = json.loads(body)
        elif isinstance(body, dict):
            js = body
        if isinstance(js, dict):
            msg = js.get("Information") or js.get("Note") or js.get("Error Message")
            if msg and len(js) <= 2:
                m = _redact(msg, self.key)
                low = m.lower()
                if "premium" in low:
                    self._record(function, symbol, "PREMIUM_ONLY")
                    raise PremiumOnly(m[:200])
                if "rate limit" in low or "per day" in low or "requests per" in low or "frequency" in low:
                    self._exhausted = True
                    self._record(function, symbol, "RATE_LIMIT")
                    raise QuotaExhausted(m[:200])
                self._record(function, symbol, "ERROR")
                raise AlphaVantageError(m[:200])
            self._record(function, symbol, "OK")
            return js
        self._record(function, symbol, "OK")
        return body

    # ---- fungsi tingkat tinggi
    def earnings(self, ticker: str) -> pd.DataFrame:
        return parse_earnings(self.call("EARNINGS", symbol=ticker), ticker)

    def earnings_calendar(self, horizon: str = "3month") -> pd.DataFrame:
        return parse_calendar(self.call("EARNINGS_CALENDAR", horizon=horizon))

    def treasury(self, maturity: str) -> pd.DataFrame:
        return parse_series(self.call("TREASURY_YIELD", interval="daily", maturity=maturity), f"UST{maturity.upper()}")

    def fed_funds(self) -> pd.DataFrame:
        return parse_series(self.call("FEDERAL_FUNDS_RATE", interval="daily"), "FEDFUNDS")

    def news(self, time_from: pd.Timestamp, time_to: pd.Timestamp, limit: int = 1000) -> pd.DataFrame:
        js = self.call("NEWS_SENTIMENT", time_from=time_from.strftime("%Y%m%dT%H%M"),
                       time_to=time_to.strftime("%Y%m%dT%H%M"), sort="LATEST", limit=str(limit))
        return parse_news(js, float(get(self.cfg, "alpha_vantage.news_min_relevance", 0.25) or 0.25))
