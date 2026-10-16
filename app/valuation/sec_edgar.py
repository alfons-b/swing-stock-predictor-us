"""SECEdgarFundamentalProvider — laporan keuangan resmi emiten AS dari SEC EDGAR (gratis, point-in-time).

Sumber (API publik SEC, tanpa API key):
- https://www.sec.gov/files/company_tickers.json            → pemetaan ticker → CIK
- https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json → semua fakta XBRL per emiten

Mengapa ini point-in-time ASLI (berbeda dari snapshot Yahoo):
- setiap fakta membawa `filed` (tanggal laporan 10-K/10-Q diterima SEC) dan `accn` (nomor filing);
- angka sebuah periode diambil dari filing yang MELAPORKAN periode itu pertama kali (bukan angka restated
  dari filing tahun berikutnya). `publication_date` = tanggal `filed` filing tersebut;
- kuartal yang tidak dilaporkan terpisah (arus kas 10-Q bersifat year-to-date; Q4 tidak punya 10-Q) diturunkan
  dari selisih YTD yang SUDAH diketahui pada tanggal filing: Qn = YTD(n) − YTD(n−1); Q4 = FY − YTD 9 bulan.

Ketentuan SEC (https://www.sec.gov/os/accessing-edgar-data): wajib header User-Agent yang menyebut nama & email
pemilik, maksimal 10 request/detik. Isi env `SEC_USER_AGENT`, mis. "Budi Santoso budi@email.com". Tanpa itu
provider berstatus NOT_CONFIGURED dan sistem memakai Yahoo (snapshot) sebagai cadangan.

Batasan yang dinyatakan terus terang:
- hanya formulir 10-K dan 10-Q (emiten asing pelapor 20-F/40-F tidak tercakup → fallback Yahoo);
- amandemen (10-K/A, 10-Q/A) tidak dipakai: angka asli yang dipakai pasar saat itu lebih relevan untuk backtest;
- nama tag XBRL berbeda antar emiten/tahun — dipakai daftar tag cadangan; tag yang tidak ada → None (bukan nol).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from app.config import get, resolve_path
from app.data.tickers import normalize_ticker
from app.database.repository import now_utc
from app.utils.logging_utils import get_logger
from app.valuation.financial_statement_normalizer import CANONICAL, Statement

log = get_logger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"

# field kanonik → tag us-gaap (urutan = prioritas). Nilai pertama yang tersedia di filing yang sama dipakai.
DURATION_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                "RevenuesNetOfInterestExpense", "TotalRevenuesAndOtherIncome"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "ebitda": [],
}
INSTANT_TAGS = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "total_equity": ["StockholdersEquity",
                     "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "total_debt": ["LongTermDebt", "DebtLongtermAndShorttermCombinedAmount", "LongTermDebtNoncurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}
SHARES_DEI = ["EntityCommonStockSharesOutstanding"]
SHARES_GAAP = ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"]
FORMS = {"10-K", "10-Q"}


def _unit_rows(fact: dict, unit_pref=("USD", "USD/shares", "shares")) -> list[dict]:
    units = fact.get("units") or {}
    for u in unit_pref:
        if u in units:
            return units[u]
    return next(iter(units.values()), []) if units else []


def _days(start, end) -> int | None:
    if not start or not end:
        return None
    return (pd.Timestamp(end) - pd.Timestamp(start)).days


def _label_end(end: pd.Timestamp) -> pd.Timestamp:
    """Tahun fiskal 52/53 minggu sering berakhir di hari-hari awal bulan (mis. 1 Jan) → label pakai akhir bulan
    sebelumnya, agar Q4 tidak terbaca sebagai Q1."""
    end = pd.Timestamp(end)
    return (end - pd.Timedelta(days=end.day)) if end.day <= 7 else end


def _quarter_of(end: pd.Timestamp, fy_end_month: int) -> str:
    m = (_label_end(end).month - fy_end_month - 1) % 12 + 1
    return f"Q{(m + 2) // 3}"


class FactIndex:
    """Fakta companyfacts terindeks: tag → list entri 10-K/10-Q (tanpa amandemen)."""

    def __init__(self, facts: dict):
        self.gaap = (facts.get("facts") or {}).get("us-gaap") or {}
        self.dei = (facts.get("facts") or {}).get("dei") or {}
        self._cache: dict[str, list[dict]] = {}

    def rows(self, tag: str, ns: str = "gaap") -> list[dict]:
        key = f"{ns}:{tag}"
        if key not in self._cache:
            src = self.gaap if ns == "gaap" else self.dei
            fact = src.get(tag)
            out = []
            if fact:
                for r in _unit_rows(fact):
                    if r.get("form") in FORMS and r.get("val") is not None and r.get("filed"):
                        out.append(r)
            self._cache[key] = out
        return self._cache[key]

    def filings(self) -> pd.DataFrame:
        """Satu baris per filing (accn): form, filed, period_end.

        period_end = tanggal akhir dengan fakta us-gaap TERBANYAK di filing itu (seri: yang terbaru). Periode berjalan
        selalu punya laba rugi + neraca + arus kas, jadi menang atau seri dengan tahun pembanding; fakta
        "subsequent events" yang bertanggal setelah periode jumlahnya sedikit sehingga tidak terpilih."""
        counts: dict[tuple, int] = {}
        meta: dict[str, tuple] = {}
        for tag in self.gaap:
            for r in self.rows(tag):
                a = r["accn"]
                key = (a, pd.Timestamp(r["end"]))
                counts[key] = counts.get(key, 0) + 1
                meta.setdefault(a, (r["form"], pd.Timestamp(r["filed"])))
        best: dict[str, tuple] = {}
        for (a, end), n in counts.items():
            if a not in best or (n, end) > best[a]:
                best[a] = (n, end)
        recs = [{"accn": a, "form": meta[a][0], "filed": meta[a][1], "period_end": e} for a, (_, e) in best.items()]
        df = pd.DataFrame(recs, columns=["accn", "form", "filed", "period_end"])
        return df.sort_values(["filed", "period_end"]).reset_index(drop=True) if len(df) else df

    def value_in_filing(self, tags: list[str], accn: str, end: pd.Timestamp, days_lo=None, days_hi=None,
                        start=None) -> float | None:
        """Nilai fakta dari filing `accn` untuk periode yang berakhir di `end` (durasi dalam [lo, hi] hari)."""
        for tag in tags:
            for r in self.rows(tag):
                if r["accn"] != accn or pd.Timestamp(r["end"]) != end:
                    continue
                if start is not None and (not r.get("start") or pd.Timestamp(r["start"]) != pd.Timestamp(start)):
                    continue
                d = _days(r.get("start"), r.get("end"))
                if days_lo is not None and (d is None or not (days_lo <= d <= days_hi)):
                    continue
                return float(r["val"])
        return None

    def first_reported(self, tags: list[str], start, end, known_by: pd.Timestamp) -> float | None:
        """Nilai (start, end) seperti PERTAMA KALI dilaporkan, dan hanya bila filing itu ≤ known_by."""
        best = None
        for tag in tags:
            for r in self.rows(tag):
                if r.get("start") != start or r.get("end") != end:
                    continue
                f = pd.Timestamp(r["filed"])
                if f <= known_by and (best is None or f < best[0]):
                    best = (f, float(r["val"]))
            if best is not None:
                return best[1]
        return None

    def ytd_start(self, tags: list[str], accn: str, end: pd.Timestamp) -> str | None:
        """Awal periode YTD terpanjang yang berakhir di `end` dalam filing `accn` (= awal tahun fiskal)."""
        starts = []
        for tag in tags:
            for r in self.rows(tag):
                if r["accn"] == accn and pd.Timestamp(r["end"]) == end and r.get("start"):
                    starts.append(r["start"])
        return min(starts) if starts else None


def statements_from_facts(ticker: str, facts: dict, max_years: int = 6, source: str = "sec_edgar",
                          today: pd.Timestamp | None = None) -> list[Statement]:
    """companyfacts JSON → list Statement (FY + Q1..Q4 diskret) dengan publication_date = tanggal filed."""
    ix = FactIndex(facts)
    fil = ix.filings()
    if fil.empty:
        return []
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    fil = fil[fil["period_end"] >= today - pd.DateOffset(years=max_years)]
    # filing pertama untuk setiap periode (bukan pengulangan sebagai komparatif)
    fil = fil.sort_values("filed").drop_duplicates(["form", "period_end"], keep="first")
    k_ends = fil.loc[fil["form"] == "10-K", "period_end"]
    fy_end_month = int(k_ends.map(lambda d: _label_end(d).month).mode().iloc[0]) if len(k_ends) else 12
    retrieved = now_utc()
    out: list[Statement] = []

    def instant_items(accn, end) -> dict:
        it = {k: ix.value_in_filing(tags, accn, end) for k, tags in INSTANT_TAGS.items()}
        sh = None
        for tag in SHARES_DEI:                       # tanggal sampul (setelah akhir periode) — ambil dari filing yang sama
            for r in ix.rows(tag, "dei"):
                if r["accn"] == accn:
                    sh = float(r["val"])
        if sh is None:
            sh = ix.value_in_filing(SHARES_GAAP[:1], accn, end)
        if sh is None:
            sh = ix.value_in_filing(SHARES_GAAP[1:], accn, end, 80, 100) or ix.value_in_filing(SHARES_GAAP[1:], accn, end, 350, 380)
        it["shares_outstanding"] = sh
        return it

    def finish(items: dict) -> dict:
        items = {**{k: None for k in CANONICAL}, **items}
        for k in ("capex", "dividends_paid", "interest_expense"):
            if items.get(k) is not None:
                items[k] = abs(items[k])
        if items.get("ebit") is None and items.get("operating_income") is not None:
            items["ebit"] = items["operating_income"]
        if items.get("free_cash_flow") is None and items.get("operating_cash_flow") is not None \
                and items.get("capex") is not None:
            items["free_cash_flow"] = items["operating_cash_flow"] - items["capex"]
        return items                                 # tag yang tidak ada tetap None (UNAVAILABLE), tidak ditebak

    for f in fil.itertuples():
        end, accn, filed = f.period_end, f.accn, f.filed
        if f.form == "10-K":
            fy = {k: ix.value_in_filing(tags, accn, end, 350, 380) for k, tags in DURATION_TAGS.items() if tags}
            items = finish({**fy, **instant_items(accn, end)})
            out.append(Statement(ticker=ticker, period_end=end, period_type="FY", currency="USD", items=items,
                                 source=source, retrieved_at=retrieved, publication_date=filed))
            # Q4 = FY − YTD 9 bulan (keduanya diketahui pada tanggal 10-K)
            q4 = {}
            for k, tags in DURATION_TAGS.items():
                if not tags:
                    continue
                direct = ix.value_in_filing(tags, accn, end, 80, 100)
                if direct is not None:
                    q4[k] = direct
                    continue
                start = ix.ytd_start(tags, accn, end)
                fyv = fy.get(k)
                if fyv is None or start is None:
                    q4[k] = None
                    continue
                q3_end = _prior_quarter_end(ix, tags, start, end, filed)
                ytd9 = ix.first_reported(tags, start, q3_end.strftime("%Y-%m-%d"), filed) if q3_end is not None else None
                q4[k] = None if ytd9 is None else fyv - ytd9
            if any(q4.get(k) is not None for k in ("revenue", "net_income")):
                out.append(Statement(ticker=ticker, period_end=end, period_type="Q4", currency="USD",
                                     items=finish({**q4, **instant_items(accn, end)}), source=source,
                                     retrieved_at=retrieved, publication_date=filed))
        else:  # 10-Q
            q = {}
            for k, tags in DURATION_TAGS.items():
                if not tags:
                    continue
                direct = ix.value_in_filing(tags, accn, end, 80, 100)
                if direct is not None:
                    q[k] = direct
                    continue
                start = ix.ytd_start(tags, accn, end)
                ytd = ix.value_in_filing(tags, accn, end, 150, 290, start=start) if start else None
                if ytd is None:
                    q[k] = None
                    continue
                prev_end = _prior_quarter_end(ix, tags, start, end, filed)
                prev = ix.first_reported(tags, start, prev_end.strftime("%Y-%m-%d"), filed) if prev_end is not None else None
                q[k] = None if prev is None else ytd - prev
            if not any(q.get(k) is not None for k in ("revenue", "net_income", "operating_cash_flow")):
                continue
            out.append(Statement(ticker=ticker, period_end=end, period_type=_quarter_of(end, fy_end_month),
                                 currency="USD", items=finish({**q, **instant_items(accn, end)}), source=source,
                                 retrieved_at=retrieved, publication_date=filed))
    return out


def _prior_quarter_end(ix: FactIndex, tags: list[str], start: str, end: pd.Timestamp, known_by) -> pd.Timestamp | None:
    """Akhir periode YTD sebelumnya (awal sama, berakhir ±3 bulan sebelum `end`) yang sudah dilaporkan."""
    best = None
    for tag in tags:
        for r in ix.rows(tag):
            if r.get("start") != start or pd.Timestamp(r["filed"]) > known_by:
                continue
            e = pd.Timestamp(r["end"])
            gap = (end - e).days
            if 75 <= gap <= 105 and (best is None or e > best):
                best = e
    return best


class SECEdgarFundamentalProvider:
    """Adapter untuk app/valuation/fundamental_provider.py (antarmuka FundamentalDataProvider)."""
    name = "sec_edgar"
    point_in_time = True

    def __init__(self, entry: dict, cfg: dict):
        self.entry, self.cfg = entry, cfg
        self.name = entry.get("name", self.name)
        self.priority = int(entry.get("priority", 1))
        self.user_agent = (entry.get("user_agent") or os.environ.get("SEC_USER_AGENT") or "").strip()
        self.max_years = int(entry.get("max_years", 6))
        self.min_interval = 1.0 / float(entry.get("max_requests_per_second", 8))
        self.cache = resolve_path(cfg, str(get(cfg, "project.cache_dir", "data/cache"))) / "sec"
        self._ciks: dict[str, int] | None = None
        self._last = 0.0

    def availability(self):
        if not self.user_agent or "@" not in self.user_agent:
            return "NOT_CONFIGURED", ("SEC_USER_AGENT belum diisi (format: 'Nama Anda email@domain.com'). SEC mewajibkan "
                                      "header ini; tanpa itu dipakai cadangan Yahoo (snapshot, bukan point-in-time).")
        return "AVAILABLE", "SEC EDGAR companyfacts (10-K/10-Q, point-in-time dari tanggal filed)"

    def _get_json(self, url: str):
        import requests
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        last_err = None
        for i in range(3):
            self._last = time.monotonic()
            try:
                r = requests.get(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                                 timeout=60)
            except Exception as e:
                last_err = e
                time.sleep(2 ** i)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 503):
                last_err = RuntimeError(f"SEC HTTP {r.status_code}")
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise last_err or RuntimeError("SEC request gagal")

    def cik_map(self) -> dict[str, int]:
        if self._ciks is None:
            self.cache.mkdir(parents=True, exist_ok=True)
            p = self.cache / "company_tickers.json"
            data = None
            if p.exists() and time.time() - p.stat().st_mtime < 7 * 86400:
                data = json.loads(p.read_text())
            if data is None:
                data = self._get_json(TICKERS_URL) or {}
                p.write_text(json.dumps(data))
            m = {}
            for rec in (data.values() if isinstance(data, dict) else data):
                try:
                    m[normalize_ticker(rec["ticker"])] = int(rec["cik_str"])
                except (ValueError, KeyError, TypeError):
                    continue
            self._ciks = m
        return self._ciks

    def get_statements(self, ticker: str) -> list[Statement]:
        t = normalize_ticker(ticker)
        cik = self.cik_map().get(t)
        if cik is None:
            return []
        facts = self._get_json(FACTS_URL.format(cik=cik))
        if not facts:
            return []
        return statements_from_facts(t, facts, self.max_years, self.name)


def earnings_from_submissions(blocks: list[dict], since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Blok `filings.recent` / halaman lama submissions SEC → 8-K berisi Item 2.02 (Results of Operations)."""
    from app.features.earnings import reaction_date
    rows = []
    for b in blocks:
        forms = b.get("form") or []
        for i, f in enumerate(forms):
            if f not in ("8-K", "8-K/A"):
                continue
            items = str((b.get("items") or [""] * len(forms))[i] or "")
            if "2.02" not in items.split(","):
                continue
            rows.append({"accession": b["accessionNumber"][i], "filing_date": b["filingDate"][i],
                         "accepted_at": (b.get("acceptanceDateTime") or [None] * len(forms))[i], "form": f})
    df = pd.DataFrame(rows, columns=["accession", "filing_date", "accepted_at", "form"])
    if df.empty:
        return df.assign(reaction_date=pd.Series(dtype="datetime64[ns]"))
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    if since is not None:
        df = df[df["filing_date"] >= since]
    df = df[df["form"] == "8-K"]                     # amandemen (8-K/A) bukan pengumuman baru
    df["reaction_date"] = reaction_date(df["accepted_at"], df["filing_date"])
    return df.drop_duplicates("reaction_date").sort_values("reaction_date").reset_index(drop=True)


def _fetch_earnings(self, ticker: str, years: int = 7) -> pd.DataFrame:
    """Riwayat tanggal rilis laba (8-K Item 2.02). Halaman lama diambil hanya sejauh yang dibutuhkan."""
    t = normalize_ticker(ticker)
    cik = self.cik_map().get(t)
    if cik is None:
        return pd.DataFrame()
    sub = self._get_json(SUBMISSIONS_URL.format(cik=cik)) or {}
    filings = sub.get("filings") or {}
    blocks = [filings.get("recent") or {}]
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    for f in filings.get("files") or []:
        try:
            if pd.Timestamp(f.get("filingTo")) < since:
                continue
        except (TypeError, ValueError):
            pass
        page = self._get_json(SUBMISSIONS_PAGE_URL.format(name=f["name"]))
        if page:
            blocks.append(page)
    ev = earnings_from_submissions(blocks, since)
    ev.insert(0, "ticker", t)
    return ev


SECEdgarFundamentalProvider.get_earnings_events = _fetch_earnings


def load_facts_file(path: str | Path) -> dict:
    """Utilitas test/debug: baca companyfacts JSON dari file."""
    return json.loads(Path(path).read_text())
