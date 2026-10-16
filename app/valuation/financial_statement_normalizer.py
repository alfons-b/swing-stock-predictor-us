"""Normalisasi laporan keuangan ke field kanonik + pemeriksaan kualitas.

Prinsip:
- Field yang tidak tersedia = None (UNAVAILABLE), BUKAN nol.
- Satuan: nilai mata uang dalam satuan penuh mata uang laporan (`currency`), lembar saham dalam lembar.
- Konvensi tanda dibuat konsisten: capex dan dividen dibayar disimpan sebagai angka POSITIF (arus kas keluar).
- Setiap laporan membawa provenance: periode, tipe periode, tanggal publikasi (bila diketahui),
  first_known_date (kapan data boleh dipakai), sumber, waktu pengambilan.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import pandas as pd

CANONICAL = [
    # laba rugi
    "revenue", "gross_profit", "operating_income", "ebit", "ebitda", "net_income", "interest_expense", "tax_expense",
    "pretax_income", "eps_diluted",
    # neraca
    "total_assets", "total_liabilities", "total_equity", "total_debt", "cash", "current_assets", "current_liabilities",
    "shares_outstanding",
    # arus kas
    "operating_cash_flow", "capex", "free_cash_flow", "dividends_paid",
]
FLOW_ITEMS = {"revenue", "gross_profit", "operating_income", "ebit", "ebitda", "net_income", "interest_expense",
              "tax_expense", "pretax_income", "eps_diluted", "operating_cash_flow", "capex", "free_cash_flow", "dividends_paid"}
STOCK_ITEMS = set(CANONICAL) - FLOW_ITEMS

# Alias nama baris (Yahoo Finance / ekspor umum / Bahasa Indonesia). Urutan = prioritas.
ALIASES = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue", "Pendapatan", "Pendapatan Usaha", "Penjualan"],
    "gross_profit": ["Gross Profit", "Laba Bruto"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported", "Laba Usaha"],
    "ebit": ["EBIT"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income Common Stockholders", "Net Income", "Net Income Including Noncontrolling Interests",
                   "Laba Bersih", "Laba Tahun Berjalan Yang Dapat Diatribusikan Kepada Pemilik Entitas Induk"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating", "Beban Bunga"],
    "tax_expense": ["Tax Provision", "Beban Pajak"],
    "pretax_income": ["Pretax Income", "Laba Sebelum Pajak"],
    "eps_diluted": ["Diluted EPS", "Basic EPS", "EPS", "Laba Per Saham"],
    "total_assets": ["Total Assets", "Jumlah Aset", "Total Aset"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities", "Jumlah Liabilitas"],
    "total_equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest",
                     "Jumlah Ekuitas Yang Dapat Diatribusikan Kepada Pemilik Entitas Induk", "Jumlah Ekuitas"],
    "total_debt": ["Total Debt", "Jumlah Utang Berbunga"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Kas Dan Setara Kas"],
    "current_assets": ["Current Assets", "Jumlah Aset Lancar"],
    "current_liabilities": ["Current Liabilities", "Jumlah Liabilitas Jangka Pendek"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued", "Jumlah Saham Beredar"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                            "Arus Kas Bersih Dari Aktivitas Operasi"],
    "capex": ["Capital Expenditure", "Purchase Of PPE", "Belanja Modal"],
    "free_cash_flow": ["Free Cash Flow"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid", "Dividen Dibayar"],
}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else x


@dataclass
class Statement:
    ticker: str
    period_end: pd.Timestamp
    period_type: str                      # FY | Q1 | Q2 | Q3 | Q4
    currency: str | None
    items: dict
    source: str
    retrieved_at: str
    publication_date: pd.Timestamp | None = None
    first_known_date: pd.Timestamp | None = None
    quality_status: str = "OK"
    quality_notes: list = field(default_factory=list)

    @property
    def fiscal_year(self) -> int:
        return int(pd.Timestamp(self.period_end).year)

    def items_hash(self) -> str:
        blob = json.dumps({k: self.items.get(k) for k in CANONICAL}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pick(raw: dict, field_name: str):
    for alias in ALIASES.get(field_name, []):
        if alias in raw:
            v = _num(raw[alias])
            if v is not None:
                return v
    return None


def normalize_items(raw: dict) -> dict:
    """raw: {nama baris: nilai} → field kanonik. Tanda dibuat konsisten; FCF dihitung bila komponen ada."""
    it = {k: pick(raw, k) for k in CANONICAL}
    for k in ("capex", "dividends_paid", "interest_expense"):
        if it[k] is not None:
            it[k] = abs(it[k])                 # Yahoo melaporkan arus kas keluar sebagai negatif
    if it["free_cash_flow"] is None and it["operating_cash_flow"] is not None and it["capex"] is not None:
        it["free_cash_flow"] = it["operating_cash_flow"] - it["capex"]
    if it["ebit"] is None and it["operating_income"] is not None:
        it["ebit"] = it["operating_income"]
    return it


def quality_checks(st: Statement, price_currency: str = "USD") -> Statement:
    """Pemeriksaan kualitas laporan. Tidak mengubah angka — hanya memberi status & catatan."""
    it, notes = st.items, []
    core = ["revenue", "net_income", "total_equity"]
    missing = [k for k in core if it.get(k) is None]
    if missing:
        notes.append("field inti tidak tersedia: " + ", ".join(missing))
    if it.get("revenue") is not None and it["revenue"] < 0:
        notes.append("revenue negatif")
    ta, tl, te = it.get("total_assets"), it.get("total_liabilities"), it.get("total_equity")
    if None not in (ta, tl, te) and ta > 0 and abs(ta - tl - te) / ta > 0.05:
        notes.append("aset ≠ liabilitas + ekuitas (>5%) — kemungkinan kepentingan nonpengendali/klasifikasi berbeda")
    eps, ni, sh = it.get("eps_diluted"), it.get("net_income"), it.get("shares_outstanding")
    if None not in (eps, ni, sh) and sh > 0 and ni != 0:
        implied = ni / sh
        if eps != 0 and not (0.5 < implied / eps < 2.0):
            notes.append(f"EPS {eps:.4g} tidak konsisten dengan laba/saham {implied:.4g} — cek satuan")
    if st.currency and price_currency and st.currency.upper() != price_currency.upper():
        notes.append(f"mata uang laporan {st.currency} ≠ mata uang harga {price_currency} — butuh kurs (FX)")
    st.quality_notes = notes
    if missing and len(missing) == len(core):
        st.quality_status = "INSUFFICIENT"
    elif any("tidak konsisten" in n or "mata uang" in n for n in notes):
        st.quality_status = "CHECK"
    elif notes:
        st.quality_status = "PARTIAL"
    else:
        st.quality_status = "OK"
    return st
