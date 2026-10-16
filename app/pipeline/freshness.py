"""Data freshness (§23): UP_TO_DATE | UPDATE_REQUIRED | STALE | UNAVAILABLE.

Rekomendasi BUY hanya boleh keluar bila data tidak STALE/UNAVAILABLE. Setiap rekomendasi selalu
diberi label tanggal data yang dipakai — tidak pernah disajikan seolah-olah data terbaru.
"""
from __future__ import annotations

import pandas as pd

from app.config import get
from app.database.repository import Repository
from app.market.calendar import TradingCalendar

ALLOW_TRADING = {"UP_TO_DATE", "UPDATE_REQUIRED"}


def data_freshness(cfg: dict, repo: Repository, cal: TradingCalendar, provider_latest=None) -> dict:
    expected = cal.expected_latest_market_date()
    db_latest = repo.max_price_date()
    stale_after = int(get(cfg, "calendar.stale_after_trading_days", 2))
    if db_latest is None:
        status, lag = "UNAVAILABLE", None
    else:
        lag = cal.trading_days_between(db_latest, expected)
        if lag == 0:
            status = "UP_TO_DATE"
        elif lag > stale_after:
            status = "STALE"
        else:
            status = "UPDATE_REQUIRED"
    return {"status": status, "latest_expected_market_date": str(expected.date()),
            "latest_database_date": None if db_latest is None else str(db_latest.date()),
            "latest_provider_date": None if provider_latest is None else str(pd.Timestamp(provider_latest).date()),
            "lag_trading_days": lag, "trading_allowed": status in ALLOW_TRADING}
