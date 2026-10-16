"""ProviderChain (§22): provider diurutkan berdasarkan priority; setiap batch dicoba dengan retry +
exponential backoff; ticker yang tetap gagal dicoba di provider berikutnya. Satu ticker gagal
TIDAK membatalkan pipeline — ia dilaporkan di `failed`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from app.config import get
from app.data.providers.base import MarketDataProvider, ProviderUnavailable
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def _truthy(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on") if not isinstance(v, bool) else v


def make_providers(cfg: dict) -> list[MarketDataProvider]:
    from app.data.providers.csv_provider import CSVProvider
    from app.data.providers.yahoo_provider import YahooFinanceProvider
    classes = {"csv": CSVProvider, "yahoo": YahooFinanceProvider}
    out = []
    for e in get(cfg, "market_data.providers", []):
        if not _truthy(e.get("enabled", True)):
            continue
        if e["type"] not in classes:
            log.warning("Tipe provider tidak dikenal: %s", e["type"])
            continue
        try:
            out.append(classes[e["type"]](e, cfg))
        except ProviderUnavailable as ex:  # mis. cadangan CSV tanpa data di runner cloud — normal, tidak dipakai
            log.info("Provider %s dilewati: %s", e.get("name"), ex)
        except Exception as ex:
            log.warning("Provider %s tidak bisa diinisialisasi: %s", e.get("name"), ex)
    return sorted(out, key=lambda p: p.priority)


@dataclass
class FetchResult:
    frames: dict = field(default_factory=dict)       # ticker → DataFrame
    sources: dict = field(default_factory=dict)      # ticker → nama provider
    adjusted: dict = field(default_factory=dict)     # ticker → bool
    failed: dict = field(default_factory=dict)       # ticker → pesan error
    provider_stats: dict = field(default_factory=dict)


class ProviderChain:
    def __init__(self, cfg: dict, providers: list[MarketDataProvider] | None = None, sleep=time.sleep):
        self.cfg = cfg
        self.providers = providers if providers is not None else make_providers(cfg)
        self.attempts = int(get(cfg, "market_data.retry.attempts", 3))
        self.backoff = float(get(cfg, "market_data.retry.backoff_seconds", 2))
        self.batch = int(get(cfg, "market_data.batch_size", 50))
        self.sleep = sleep
        if not self.providers:
            raise RuntimeError("Tidak ada market data provider yang aktif (config/sources.yaml)")

    @property
    def is_synthetic(self) -> bool:
        return any(getattr(p, "is_synthetic", False) for p in self.providers)

    def _retry(self, fn, what: str):
        last = None
        for i in range(self.attempts):
            try:
                return fn()
            except ProviderUnavailable:
                raise
            except Exception as e:
                last = e
                if i < self.attempts - 1:
                    wait = self.backoff * (2 ** i)
                    log.warning("%s gagal (%s), retry %d/%d dalam %.0fs", what, type(e).__name__, i + 1, self.attempts - 1, wait)
                    self.sleep(wait)
        raise last

    def fetch_prices(self, requests: dict[str, pd.Timestamp], end) -> FetchResult:
        """requests: ticker → tanggal mulai yang dibutuhkan. Ticker dengan start sama digabung per batch."""
        res = FetchResult()
        pending = dict(requests)
        for prov in self.providers:
            if not pending:
                break
            stats = {"requested": len(pending), "ok": 0, "rows": 0, "errors": 0}
            by_start: dict = {}
            for t, s in pending.items():
                by_start.setdefault(pd.Timestamp(s).normalize(), []).append(t)
            for start, tickers in by_start.items():
                for i in range(0, len(tickers), self.batch):
                    chunk = tickers[i:i + self.batch]
                    try:
                        frames = self._retry(lambda: prov.get_prices_batch(chunk, start, end), f"{prov.name} batch {len(chunk)}")
                    except ProviderUnavailable as e:
                        log.info("Provider %s tidak tersedia: %s", prov.name, e)
                        stats["errors"] += 1
                        for t in chunk:
                            if not res.failed.get(t, "").endswith("tidak ada data baru"):
                                res.failed[t] = str(e)
                        break
                    except Exception as e:
                        stats["errors"] += 1
                        for t in chunk:
                            if not res.failed.get(t, "").endswith("tidak ada data baru"):
                                res.failed[t] = f"{prov.name}: {type(e).__name__}: {e}"[:300]
                        continue
                    for t in chunk:
                        f = frames.get(t)
                        if f is not None and len(f):
                            res.frames[t], res.sources[t], res.adjusted[t] = f, prov.name, prov.returns_adjusted
                            res.failed.pop(t, None)
                            stats["ok"] += 1
                            stats["rows"] += len(f)
                        else:
                            res.failed.setdefault(t, f"{prov.name}: tidak ada data baru")
            res.provider_stats[prov.name] = stats
            # ticker yang belum dapat data → coba provider berikutnya
            pending = {t: s for t, s in pending.items() if t not in res.frames}
        return res

    def fetch_index(self, start, end) -> tuple[pd.DataFrame, str]:
        errors = []
        for prov in self.providers:
            try:
                df = self._retry(lambda: prov.get_index(start, end), f"{prov.name} index")
                if df is not None:
                    return df, prov.name
            except Exception as e:
                errors.append(f"{prov.name}: {type(e).__name__}")
        raise RuntimeError("Semua provider gagal untuk indeks acuan (S&P 500): " + "; ".join(errors))

    def fetch_actions(self, ticker: str, start, end) -> pd.DataFrame:
        for prov in self.providers:
            try:
                return prov.get_actions(ticker, start, end)
            except (NotImplementedError, ProviderUnavailable):
                continue
            except Exception as e:
                log.debug("actions %s via %s gagal: %s", ticker, prov.name, e)
        return pd.DataFrame(columns=["ticker", "ex_date", "action", "ratio", "amount"])

    def latest_available_date(self):
        for prov in self.providers:
            try:
                d = prov.latest_available_date()
                if d is not None:
                    return pd.Timestamp(d), prov.name
            except Exception:
                continue
        return None, None
