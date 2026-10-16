"""CSVProvider — file lokal dengan skema standar (ekspor feed berlisensi / dataset contoh).

`as_of` (dari cfg["_as_of"]) membatasi data yang 'terlihat' → mensimulasikan hari-hari berjalan
untuk test incremental update dan failure recovery.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from app.config import resolve_path
from app.data.providers.base import MarketDataProvider, ProviderError, ProviderUnavailable
from app.data.schema import conform_prices, conform_universe
from app.data.tickers import normalize_ticker


@lru_cache(maxsize=8)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


class CSVProvider(MarketDataProvider):
    type = "csv"

    def __init__(self, entry, cfg):
        super().__init__(entry, cfg)
        self.dir = resolve_path(cfg, str(entry.get("dir", "data/raw/sample")))
        if not (self.dir / "prices.csv").exists():
            raise ProviderUnavailable(f"folder CSV {self.dir} tidak berisi prices.csv")
        meta = self.dir / "metadata.json"
        self.metadata = json.loads(meta.read_text()) if meta.exists() else {}
        self.is_synthetic = bool(self.metadata.get("synthetic", False))
        self.returns_adjusted = bool(self.metadata.get("adjusted", False))
        self.as_of = pd.Timestamp(cfg["_as_of"]) if cfg.get("_as_of") else None

    def _read(self, name: str, required: bool = True) -> pd.DataFrame:
        p = self.dir / name
        if not p.exists():
            if required:
                raise ProviderError(f"File tidak ditemukan: {p}")
            return pd.DataFrame()
        return _read_csv(str(p), p.stat().st_mtime).copy()

    def _cut(self, df: pd.DataFrame, col: str, start=None, end=None) -> pd.DataFrame:
        d = pd.to_datetime(df[col])
        m = pd.Series(True, index=df.index)
        if start is not None:
            m &= d >= pd.Timestamp(start)
        if end is not None:
            m &= d <= pd.Timestamp(end)
        if self.as_of is not None:
            m &= d <= self.as_of
        return df[m]

    def get_universe(self):
        u = self._read("universe.csv")
        u["ticker"] = u["ticker"].map(normalize_ticker)
        if self.as_of is not None and "listing_date" in u:
            u = u[pd.to_datetime(u["listing_date"]).fillna(pd.Timestamp("1900-01-01")) <= self.as_of]
        return conform_universe(u)

    def _all_prices(self):
        p = self._read("prices.csv")
        p["ticker"] = p["ticker"].map(normalize_ticker)
        return p

    def get_prices(self, ticker, start, end):
        p = self._all_prices()
        p = self._cut(p[p["ticker"] == normalize_ticker(ticker)], "date", start, end)
        return conform_prices(p) if len(p) else pd.DataFrame(columns=["ticker", "date"])

    def get_prices_batch(self, tickers, start, end):
        p = self._all_prices()
        p = self._cut(p[p["ticker"].isin(set(tickers))], "date", start, end)
        p = conform_prices(p) if len(p) else p
        return {t: g for t, g in p.groupby("ticker")} if len(p) else {}

    def get_index(self, start, end):
        i = self._cut(self._read("index.csv"), "date", start, end).copy()
        i["date"] = pd.to_datetime(i["date"])
        return i[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    def get_actions(self, ticker, start, end):
        ca = self._read("corporate_actions.csv", required=False)
        if ca.empty:
            return pd.DataFrame(columns=["ticker", "ex_date", "action", "ratio", "amount"])
        ca["ticker"] = ca["ticker"].map(normalize_ticker)
        ca = self._cut(ca[ca["ticker"] == normalize_ticker(ticker)], "ex_date", start, end).copy()
        ca["ex_date"] = pd.to_datetime(ca["ex_date"])
        ca["action"] = ca["action"].str.upper()
        return ca

    def latest_available_date(self):
        ix = self.get_index(None, None)
        return pd.Timestamp(ix["date"].max()) if len(ix) else None

    def enrich_universe(self, tickers, limit=None):
        u = self.get_universe()
        return u[u["ticker"].isin(tickers)][["ticker", "name", "sector", "subsector"]]
