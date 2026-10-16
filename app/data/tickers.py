"""Normalisasi ticker saham AS.

Format internal = format Yahoo Finance: huruf besar, kelas saham dipisah tanda minus (`BRK-B`, `BF-B`).
Format lain yang diterima dan dinormalkan:
  `brk.b`, `BRK/B`, `BRK B`, `NYSE:BRK.B`, `AAPL.US`, `$AAPL`  →  `BRK-B` / `AAPL`
Sumber yang berbeda menulis kelas saham berbeda-beda (Wikipedia/S&P: `BRK.B`, Nasdaq Trader: `BRK.B`,
Yahoo: `BRK-B`, FINRA: `BRK.B`) — semuanya bertemu di satu format internal agar histori tidak terpecah.
"""
from __future__ import annotations

import re

# 1–5 huruf (Nasdaq/NYSE), opsional kelas saham 1–2 huruf. Angka diizinkan di root untuk kasus langka.
_VALID = re.compile(r"^[A-Z][A-Z0-9]{0,5}(-[A-Z]{1,2})?$")
_EXCH_PREFIX = re.compile(r"^(NYSE|NASDAQ|NYSEARCA|NYSEAMERICAN|AMEX|BATS|CBOE|ARCA|US)\s*:\s*")


def normalize_ticker(raw: str) -> str:
    """Ubah berbagai format menjadi format internal (`BRK-B`, `AAPL`)."""
    if raw is None:
        raise ValueError("Ticker kosong")
    t = str(raw).strip().upper().lstrip("$")
    t = _EXCH_PREFIX.sub("", t)
    if t.endswith(".US"):
        t = t[:-3]
    t = re.sub(r"[./\s]+", "-", t).strip("-")
    if not _VALID.match(t):
        raise ValueError(f"Ticker tidak valid: {raw!r}")
    return t


def to_provider_symbol(ticker: str, suffix: str = "") -> str:
    """Yahoo memakai format internal apa adanya (suffix kosong untuk pasar AS)."""
    return f"{normalize_ticker(ticker)}{suffix}"


def from_provider_symbol(symbol: str) -> str:
    return normalize_ticker(symbol)


def to_dot_symbol(ticker: str) -> str:
    """Format bertitik (`BRK.B`) yang dipakai FINRA, SEC, dan S&P."""
    return normalize_ticker(ticker).replace("-", ".")
