"""MarketDataProvider (§9). Semua provider mengembalikan skema standar (app/data/schema.py)."""
from __future__ import annotations

import pandas as pd


class ProviderError(RuntimeError):
    pass


class ProviderUnavailable(ProviderError):
    """Provider tidak bisa dipakai (credentials/subscription tidak ada) → langsung fallback, tanpa retry."""


class MarketDataProvider:
    name = "base"
    type = "base"
    #: harga sudah disesuaikan split oleh provider (Yahoo) → histori harus diunduh ulang bila ada split baru
    returns_adjusted = False
    is_synthetic = False

    def __init__(self, entry: dict, cfg: dict):
        self.entry = entry
        self.cfg = cfg
        self.name = entry.get("name", self.type)
        self.priority = int(entry.get("priority", 50))

    def get_universe(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_prices(self, ticker: str, start, end) -> pd.DataFrame:
        raise NotImplementedError

    def get_actions(self, ticker: str, start, end) -> pd.DataFrame:
        raise NotImplementedError

    def get_index(self, start, end) -> pd.DataFrame:
        raise NotImplementedError

    def get_prices_batch(self, tickers: list[str], start, end) -> dict[str, pd.DataFrame]:
        """Default: per ticker. Provider yang mendukung batch (Yahoo) meng-override ini."""
        out = {}
        for t in tickers:
            out[t] = self.get_prices(t, start, end)
        return out

    def latest_available_date(self):
        ix = self.get_index(pd.Timestamp.today() - pd.Timedelta(days=15), None)
        return pd.Timestamp(ix["date"].max()) if len(ix) else None

    def enrich_universe(self, tickers: list[str], limit: int | None = None) -> pd.DataFrame:
        """Opsional: nama/sektor untuk ticker yang belum punya metadata."""
        return pd.DataFrame(columns=["ticker", "name", "sector"])
