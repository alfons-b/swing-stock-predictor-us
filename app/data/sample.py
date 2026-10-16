"""Generator dataset CONTOH (sintetis) untuk menguji pipeline end-to-end tanpa internet.

PERINGATAN: hasil apa pun dari data ini BUKAN rekomendasi saham. Ticker fiktif (awalan Z),
`metadata.json` berisi `synthetic: true`, dan seluruh output scan akan diberi peringatan.

Dataset sengaja memuat kasus sulit untuk menguji data quality:
- ZSUS: suspensi 25 hari (volume 0, harga beku)
- ZIPO: IPO baru (riwayat pendek)
- ZDEL: delisting 2022 (uji survivorship bias)
- ZSPL: stock split 1:5 yang BELUM disesuaikan di harga + entri corporate_actions.csv
- ZILL: sangat illiquid
- baris duplikat, OHLC tidak valid, dan nilai hilang
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.market_rules import round_to_tick_vec

SECTORS = {  # sektor GICS
    "Financials": "Diversified Banks", "Energy": "Oil & Gas Exploration & Production", "Materials": "Specialty Chemicals",
    "Consumer Staples": "Packaged Foods & Meats", "Communication Services": "Integrated Telecommunication Services",
    "Information Technology": "Application Software", "Health Care": "Pharmaceuticals", "Real Estate": "Industrial REITs",
}


def _nyse_holidays() -> pd.DatetimeIndex:
    """Libur NYSE dari config/holidays.yaml (tahun di luar daftar → semua hari kerja dianggap bursa)."""
    import yaml
    p = Path(__file__).resolve().parents[2] / "config" / "holidays.yaml"
    if not p.exists():
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("holidays") or []))


def _codes(n: int) -> list[str]:
    letters = "ABCDEFGHJKLMNPQRSTUVWXY"
    out = []
    for a in letters:
        for b in letters:
            out.append(f"Z{a}{b}K")
            if len(out) == n:
                return out
    return out


def generate_sample_dataset(out_dir: Path, n_tickers: int = 45, start: str = "2017-01-02",
                            end: str = "2026-10-02", seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, end)
    dates = dates[~dates.isin(_nyse_holidays())]       # tidak ada bar di hari libur NYSE (seperti data asli)
    T = len(dates)

    # --- regime pasar (Markov) → drift & vol pasar
    states = np.zeros(T, dtype=int)  # 0 bull, 1 neutral, 2 bear
    P = np.array([[0.985, 0.012, 0.003], [0.010, 0.980, 0.010], [0.004, 0.016, 0.980]])
    for t in range(1, T):
        states[t] = rng.choice(3, p=P[states[t - 1]])
    mu_m = np.array([0.0009, 0.0001, -0.0010])[states]
    sd_m = np.array([0.008, 0.009, 0.015])[states]
    mkt = mu_m + sd_m * rng.standard_t(5, T) / np.sqrt(5 / 3)

    sectors = list(SECTORS)
    sec_ret = {}
    for s in sectors:
        drift = np.zeros(T)
        for t in range(1, T):  # drift sektor persisten (momentum sektor)
            drift[t] = 0.97 * drift[t - 1] + rng.normal(0, 0.00012)
        sec_ret[s] = drift + rng.normal(0, 0.006, T)

    special = {"ZSUS": "Consumer Staples", "ZIPO": "Information Technology", "ZDEL": "Real Estate",
               "ZSPL": "Energy", "ZILL": "Health Care"}
    tickers = _codes(n_tickers - len(special)) + list(special)

    def at(frac: float) -> pd.Timestamp:  # event ditempatkan relatif terhadap rentang data
        return dates[int(frac * (T - 1))]

    ipo_date, delist_date, susp_start, split_date = at(0.88), at(0.55), at(0.66), at(0.72)
    universe, price_frames = [], []
    for i, tk in enumerate(tickers):
        sector = special.get(tk, sectors[i % len(sectors)])
        beta = rng.uniform(0.6, 1.4)
        idio_sd = rng.uniform(0.012, 0.028)
        # drift idiosinkratik persisten (OU) → memberi struktur momentum yang bisa dipelajari
        alpha = np.zeros(T)
        for t in range(1, T):
            alpha[t] = 0.985 * alpha[t - 1] + rng.normal(0, 0.00045)
        eps = rng.standard_t(4, T) / np.sqrt(2) * idio_sd
        ret = alpha + beta * mkt + sec_ret[sector] + eps
        ret = np.clip(ret, -0.24, 0.30)
        p0 = float(rng.choice([12, 25, 45, 80, 150, 300]) * rng.uniform(0.7, 1.3))   # USD
        close = p0 * np.cumprod(1 + ret)
        close = np.maximum(close, 1.0)
        gap = rng.normal(0, 0.004, T)
        open_ = np.r_[close[0], close[:-1]] * (1 + gap)
        rng_hi = np.abs(rng.normal(0, 0.009, T)) + np.maximum(0, ret) * 0.2
        rng_lo = np.abs(rng.normal(0, 0.009, T)) + np.maximum(0, -ret) * 0.2
        high = np.maximum(open_, close) * (1 + rng_hi)
        low = np.minimum(open_, close) * (1 - rng_lo)
        liq = rng.uniform(3e5, 8e6) if tk != "ZILL" else 2e4                         # saham/hari
        vol = liq * np.exp(rng.normal(0, 0.45, T)) * (1 + 18 * np.abs(ret)) * (1 + 30 * np.maximum(alpha, 0))
        df = pd.DataFrame({"ticker": tk, "date": dates, "open": open_, "high": high, "low": low,
                           "close": close, "volume": np.round(vol)})
        for c in ["open", "high", "low", "close"]:
            df[c] = round_to_tick_vec(np.maximum(df[c].to_numpy(), 1.0))
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        listing = dates[0]
        delisting = None
        if tk == "ZIPO":
            listing = ipo_date
            df = df[df["date"] >= listing]
        if tk == "ZDEL":
            delisting = delist_date
            df = df[df["date"] <= delisting]
        if tk == "ZSUS":
            m = (df["date"] >= susp_start) & (df["date"] < susp_start + pd.Timedelta(days=35))
            frozen = df.loc[m, "close"].iloc[0]
            df.loc[m, ["open", "high", "low", "close"]] = frozen
            df.loc[m, "volume"] = 0.0
        if tk == "ZSPL":  # split 1:5 — harga sebelum ex_date TIDAK disesuaikan (seperti data mentah)
            m = df["date"] < split_date
            df.loc[m, ["open", "high", "low", "close"]] *= 5
            df.loc[m, "volume"] /= 5
        df["value"] = df["close"] * df["volume"]
        df["frequency"] = np.round(df["volume"] / rng.uniform(800, 3000))
        price_frames.append(df)
        universe.append({"ticker": tk, "name": f"Sample {tk} Corp", "sector": sector, "subsector": SECTORS[sector],
                         "listing_date": listing.date(), "delisting_date": delisting.date() if delisting is not None else None,
                         "board": "S&P 500" if i % 2 == 0 else "S&P 400"})

    prices = pd.concat(price_frames, ignore_index=True)
    # --- kerusakan data yang disengaja (diuji validator/cleaner)
    dup = prices.sample(5, random_state=seed)
    bad = prices.sample(3, random_state=seed + 1).copy()
    bad["high"] = bad["low"] * 0.9
    prices = prices.drop(bad.index)
    miss = prices.sample(4, random_state=seed + 2).index
    prices.loc[miss, "close"] = np.nan
    prices = pd.concat([prices, dup, bad], ignore_index=True).sort_values(["ticker", "date"])
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices.to_csv(out_dir / "prices.csv", index=False)

    # --- indeks komposit (proxy S&P 500) dari rata-rata return
    idx_level = 2250 * np.cumprod(1 + mkt + np.mean([sec_ret[s] for s in sectors], axis=0) * 0.5)
    idx = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": idx_level})
    idx["open"] = np.r_[idx_level[0], idx_level[:-1]] * (1 + rng.normal(0, 0.002, T))
    idx["high"] = idx[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.003, T)))
    idx["low"] = idx[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.003, T)))
    idx["volume"] = rng.uniform(2e9, 5e9, T).round()
    idx[["date", "open", "high", "low", "close", "volume"]].to_csv(out_dir / "index.csv", index=False)

    pd.DataFrame(universe).to_csv(out_dir / "universe.csv", index=False)
    pd.DataFrame([
        {"ticker": "ZSPL", "ex_date": split_date.strftime("%Y-%m-%d"), "action": "SPLIT", "ratio": 5.0, "amount": None},
        {"ticker": tickers[0], "ex_date": at(0.8).strftime("%Y-%m-%d"), "action": "DIVIDEND", "ratio": None, "amount": 0.45},
    ]).to_csv(out_dir / "corporate_actions.csv", index=False)

    # fundamental point-in-time: available_date ~ 60-90 hari setelah period_end
    fund = []
    for tk in tickers:
        for pe in pd.date_range(dates[0] - pd.Timedelta(days=95), dates[-1], freq="QE"):
            fund.append({"ticker": tk, "period_end": pe.date(), "available_date": (pe + pd.Timedelta(days=int(rng.integers(55, 90)))).date(),
                         "revenue_growth": rng.normal(0.08, 0.15), "eps_growth": rng.normal(0.06, 0.3), "roe": rng.normal(0.12, 0.08),
                         "roa": rng.normal(0.05, 0.04), "debt_to_equity": abs(rng.normal(1.0, 0.9)), "net_margin": rng.normal(0.1, 0.08),
                         "pe": abs(rng.normal(14, 8))})
    pd.DataFrame(fund).to_csv(out_dir / "fundamentals.csv", index=False)

    news = []
    for _ in range(1500):
        tk = rng.choice(tickers)
        d = dates[rng.integers(0, T)]
        news.append({"ticker": tk, "published_at": (d + pd.Timedelta(hours=int(rng.integers(8, 20)))).strftime("%Y-%m-%d %H:%M"),
                     "headline": "Sample headline (synthetic)", "sentiment": float(np.clip(rng.normal(0.05, 0.5), -1, 1))})
    pd.DataFrame(news).to_csv(out_dir / "news.csv", index=False)

    (out_dir / "metadata.json").write_text(json.dumps({
        "synthetic": True, "adjusted": False, "generator_seed": seed,
        "warning": "DATA CONTOH SINTETIS — hanya untuk menguji pipeline. Bukan data pasar AS. Bukan rekomendasi.",
    }, indent=2))
    return out_dir
