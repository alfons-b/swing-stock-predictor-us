"""TradingCalendar (§24) — hari bursa NYSE/Nasdaq berdasarkan timezone America/New_York, bukan jam laptop/runner.

Pengguna di Indonesia: jam laptop (WIB) tidak dipakai. Penutupan pasar 16:00 ET = 03:00/04:00 WIB keesokan harinya.

Sumber hari libur:
1. config/holidays.yaml (libur resmi NYSE — perbarui tiap tahun dari nyse.com);
2. dipelajari dari data: hari kerja di dalam rentang data S&P 500 yang TIDAK punya bar = non-bursa.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from app.config import get, resolve_path


class TradingCalendar:
    def __init__(self, cfg: dict, known_trading_days: list | None = None):
        self.tz = ZoneInfo(get(cfg, "calendar.timezone", "America/New_York"))
        hh, mm = str(get(cfg, "calendar.data_ready_time", "18:00")).split(":")
        self.ready = time(int(hh), int(mm))
        self.holidays: set[pd.Timestamp] = set()
        p = resolve_path(cfg, get(cfg, "calendar.holidays_file", "config/holidays.yaml"))
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            self.holidays |= {pd.Timestamp(d).normalize() for d in data.get("holidays", []) or []}
        self.learned: set[pd.Timestamp] = set()
        if known_trading_days:
            self.learn_from_trading_days(known_trading_days)
        self._now_override = pd.Timestamp(cfg["_as_of"]) if cfg.get("_as_of") else None

    def learn_from_trading_days(self, days) -> None:
        d = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().unique().sort_values()
        if len(d) < 2:
            return
        weekdays = pd.bdate_range(d[0], d[-1])
        self.learned = set(weekdays.difference(d))

    def now(self) -> datetime:
        if self._now_override is not None:  # simulasi (test / backfill): akhir hari, data dianggap siap
            return datetime.combine(self._now_override.date(), time(23, 0), tzinfo=self.tz)
        return datetime.now(self.tz)

    def is_trading_day(self, d) -> bool:
        d = pd.Timestamp(d).normalize()
        return d.dayofweek < 5 and d not in self.holidays and d not in self.learned

    def previous_trading_day(self, d) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize() - pd.Timedelta(days=1)
        while not self.is_trading_day(d):
            d -= pd.Timedelta(days=1)
        return d

    def next_trading_day(self, d) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
        while not self.is_trading_day(d):
            d += pd.Timedelta(days=1)
        return d

    def expected_latest_market_date(self, now: datetime | None = None) -> pd.Timestamp:
        """Tanggal bursa terakhir yang datanya SEHARUSNYA sudah tersedia saat ini (jam New York)."""
        now = now or self.now()
        if now.tzinfo is not None:                 # mis. jam WIB → jam bursa (New York) dulu
            now = now.astimezone(self.tz)
        today = pd.Timestamp(now.date())
        if self.is_trading_day(today) and now.time() >= self.ready:
            return today
        return self.previous_trading_day(today)

    def trading_days_between(self, start, end) -> int:
        """Jumlah hari bursa di (start, end]."""
        if start is None:
            return 10 ** 6
        s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if e <= s:
            return 0
        return sum(1 for d in pd.date_range(s + pd.Timedelta(days=1), e) if self.is_trading_day(d))
