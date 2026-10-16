from abc import ABC, abstractmethod
from datetime import date
from typing import Any


class MarketDataProvider(ABC):
    """Provider contract. Concrete providers must never fabricate market data."""

    name: str

    @abstractmethod
    def get_universe(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_prices(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_actions(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        raise NotImplementedError
