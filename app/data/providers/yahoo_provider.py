"""YahooFinanceProvider — sumber gratis default untuk cloud run.

Fakta yang memengaruhi desain:
- OHLC Yahoo sudah disesuaikan split (retroaktif). Saat split baru terdeteksi, histori ticker tsb
  diunduh ulang (data.refetch_on_split) supaya tidak tercampur skala lama/baru.
- Yahoo tidak menyediakan daftar konstituen indeks → universe dari app/data/us_universe.py / universe.csv.
- Saham delisting/akuisisi biasanya hilang dari Yahoo → survivorship bias (dilaporkan validator).
"""
from __future__ import annotations

import pandas as pd

from app.data.providers.base import MarketDataProvider, ProviderError, ProviderUnavailable
from app.data.schema import conform_prices
from app.data.tickers import from_provider_symbol, to_provider_symbol


def _yf():
    try:
        import logging

        import yfinance as yf
        # yfinance mencetak ERROR untuk setiap ticker tanpa data ("possibly delisted"). Itu bukan error pipeline:
        # ticker tanpa data dicatat & diringkas sendiri oleh ingestion.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        return yf
    except ImportError as e:  # pragma: no cover
        raise ProviderUnavailable("yfinance belum terpasang (pip install yfinance)") from e


def _end_exclusive(end):
    return None if end is None else (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


class YahooFinanceProvider(MarketDataProvider):
    type = "yahoo"
    returns_adjusted = True

    def __init__(self, entry, cfg):
        super().__init__(entry, cfg)
        self.suffix = entry.get("suffix", "")
        self.index_symbol = entry.get("index_symbol", "^GSPC")

    def get_universe(self):
        raise NotImplementedError("Yahoo tidak menyediakan daftar konstituen indeks AS")

    @staticmethod
    def _frame(sub: pd.DataFrame, ticker: str) -> pd.DataFrame:
        sub = sub.dropna(how="all").reset_index()
        if sub.empty:
            return pd.DataFrame(columns=["ticker", "date"])
        sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
        sub = sub.rename(columns={"datetime": "date"})
        sub["ticker"] = ticker
        sub = sub.dropna(subset=["close"])
        return conform_prices(sub[["ticker", "date", "open", "high", "low", "close", "volume"]])

    def get_prices_batch(self, tickers, start, end):
        yf = _yf()
        syms = [to_provider_symbol(t, self.suffix) for t in tickers]
        raw = yf.download(syms, start=pd.Timestamp(start).strftime("%Y-%m-%d"), end=_end_exclusive(end),
                          auto_adjust=False, actions=False, group_by="ticker", progress=False, threads=True)
        out = {}
        if raw is None or raw.empty:
            return out
        for sym in syms:
            try:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            f = self._frame(sub, from_provider_symbol(sym))
            if len(f):
                out[from_provider_symbol(sym)] = f
        return out

    def get_prices(self, ticker, start, end):
        return self.get_prices_batch([ticker], start, end).get(ticker, pd.DataFrame(columns=["ticker", "date"]))

    def get_actions(self, ticker, start, end):
        t = _yf().Ticker(to_provider_symbol(ticker, self.suffix))
        a = t.actions
        cols = ["ticker", "ex_date", "action", "ratio", "amount"]
        if a is None or a.empty:
            return pd.DataFrame(columns=cols)
        a = a.reset_index()
        a["ex_date"] = pd.to_datetime(a.iloc[:, 0]).dt.tz_localize(None).dt.normalize()
        rows = []
        for _, r in a.iterrows():
            if r.get("Stock Splits", 0) and r["Stock Splits"] > 0:
                ratio = float(r["Stock Splits"])
                rows.append({"ticker": ticker, "ex_date": r["ex_date"], "action": "SPLIT" if ratio > 1 else "REVERSE_SPLIT",
                             "ratio": ratio, "amount": None})
            if r.get("Dividends", 0) and r["Dividends"] > 0:
                rows.append({"ticker": ticker, "ex_date": r["ex_date"], "action": "DIVIDEND", "ratio": None,
                             "amount": float(r["Dividends"])})
        df = pd.DataFrame(rows, columns=cols)
        if start is not None:
            df = df[df["ex_date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["ex_date"] <= pd.Timestamp(end)]
        return df

    def get_index(self, start, end):
        raw = _yf().download(self.index_symbol, start=None if start is None else pd.Timestamp(start).strftime("%Y-%m-%d"),
                             end=_end_exclusive(end), auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            raise ProviderError(f"Indeks {self.index_symbol} tidak tersedia dari Yahoo")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
        return raw[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])

    def enrich_universe(self, tickers, limit: int | None = None):
        """Nama & sektor dari Yahoo (taksonomi Yahoo, bukan GICS). Dibatasi per run agar cepat."""
        yf = _yf()
        rows = []
        for t in (tickers if limit is None else tickers[:limit]):
            try:
                info = yf.Ticker(to_provider_symbol(t, self.suffix)).info or {}
                rows.append({"ticker": t, "name": info.get("longName") or info.get("shortName"),
                             "sector": info.get("sector"), "subsector": info.get("industry")})
            except Exception:
                continue
        return pd.DataFrame(rows, columns=["ticker", "name", "sector", "subsector"])
