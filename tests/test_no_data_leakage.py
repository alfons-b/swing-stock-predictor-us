"""§31 — REQUIREMENT PALING PENTING. Fitur tanggal t hanya boleh memakai data <= t.

1. Truncation: fitur dari data penuh == fitur dari data yang dipotong di t.
2. Perturbation: mengubah harga/volume/indeks/berita/fundamental SETELAH t tidak mengubah fitur <= t.
3. Sanity: label memang memakai masa depan (test ini bisa gagal bila ada kebocoran).
"""
import unittest

import numpy as np
import pandas as pd

from app.data.cleaner import clean_prices
from app.data.providers.csv_provider import CSVProvider
from app.features.builder import build_features, feature_columns
from app.models.labels import add_labels
from tests.helpers import make_cfg

STRATEGY_COLS = ["res20", "sup20", "swing_low10", "atr14", "ema20", "ema50", "sma200", "is_tradeable", "market_regime",
                 "sector_score", "liquidity_score", "base_high", "base_low"]


def load_market(cfg):
    p = CSVProvider(cfg["market_data"]["providers"][0], cfg)
    u = p.get_universe()
    from app.data.schema import conform_prices
    raw = conform_prices(p._all_prices())
    ca = pd.concat([p.get_actions(t, None, None) for t in u["ticker"]], ignore_index=True)
    clean, _ = clean_prices(raw, u, ca, cfg)
    news = p._read("news.csv", required=False)
    fund = p._read("fundamentals.csv", required=False)
    return {"prices": clean, "index": p.get_index(None, None), "universe": u, "corporate_actions": ca,
            "fundamentals": fund, "news": news}


def truncate(m, t):
    o = dict(m)
    o["prices"] = m["prices"][m["prices"]["date"] <= t].copy()
    o["index"] = m["index"][m["index"]["date"] <= t].copy()
    o["fundamentals"] = m["fundamentals"][pd.to_datetime(m["fundamentals"]["available_date"]) <= t].copy()
    o["news"] = m["news"][pd.to_datetime(m["news"]["published_at"]) < t + pd.Timedelta(days=1)].copy()
    return o


def compare(a, b, cols):
    a, b = a.set_index("ticker").sort_index(), b.set_index("ticker").sort_index()
    idx = a.index.intersection(b.index)
    bad = []
    for c in cols:
        x, y = a.loc[idx, c], b.loc[idx, c]
        if x.dtype.kind not in "fiub" or y.dtype.kind not in "fiub":
            if not (x.astype(str) == y.astype(str)).all():
                bad.append(c)
        elif not np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True).all():
            bad.append(c)
    return bad


class TestNoDataLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = make_cfg()
        cls.m = load_market(cls.cfg)
        cls.full = build_features(cls.m, cls.cfg)
        cls.cols = feature_columns(cls.full)
        d = sorted(cls.full["date"].unique())
        cls.dates = [pd.Timestamp(d[i]) for i in (300, 600, len(d) - 30)]

    def test_no_label_in_features(self):
        self.assertFalse([c for c in self.cols if c.startswith("y_") or "future" in c or "fwd" in c])

    def test_truncation(self):
        for t in self.dates:
            tr = build_features(truncate(self.m, t), self.cfg)
            bad = compare(self.full[self.full["date"] == t], tr[tr["date"] == t], self.cols + STRATEGY_COLS)
            self.assertEqual(bad, [], f"Look-ahead pada {t.date()}: {bad}")

    def test_future_perturbation(self):
        t = self.dates[1]
        m = {k: v.copy() for k, v in self.m.items()}
        rng = np.random.default_rng(0)
        fut = m["prices"]["date"] > t
        for c in ("open", "high", "low", "close"):
            m["prices"].loc[fut, c] *= rng.uniform(0.3, 3.0, fut.sum())
        m["prices"].loc[fut, "volume"] *= rng.uniform(0, 10, fut.sum())
        m["index"].loc[m["index"]["date"] > t, ["open", "high", "low", "close"]] *= 0.5
        m["news"].loc[pd.to_datetime(m["news"]["published_at"]) > t + pd.Timedelta(days=1), "sentiment"] = -1.0
        m["fundamentals"].loc[pd.to_datetime(m["fundamentals"]["available_date"]) > t, "roe"] = -9.0
        p = build_features(m, self.cfg)
        for d in (t, t - pd.Timedelta(days=30)):
            d = self.full.loc[self.full["date"] <= d, "date"].max()
            bad = compare(self.full[self.full["date"] == d], p[p["date"] == d], self.cols + STRATEGY_COLS)
            self.assertEqual(bad, [], f"Fitur {d.date()} berubah karena data masa depan: {bad}")

    def test_detector_catches_injected_leak(self):
        import app.features.builder as B
        orig = B.compute_stock_features

        def leaky(g, cfg):
            r = orig(g, cfg)
            r["f_leak"] = r["close"].shift(-2) / r["close"]
            return r
        B.compute_stock_features = leaky
        try:
            t = self.dates[1]
            full = build_features(self.m, self.cfg)
            tr = build_features(truncate(self.m, t), self.cfg)
            self.assertIn("f_leak", compare(full[full["date"] == t], tr[tr["date"] == t], ["f_leak"]))
        finally:
            B.compute_stock_features = orig

    def test_labels_use_future(self):
        lab = add_labels(self.full.copy(), self.cfg)
        t = self.dates[1]
        row = lab[lab["date"] == t].iloc[0]
        nxt = lab[(lab["ticker"] == row["ticker"]) & (lab["date"] > t)].iloc[4]
        self.assertAlmostEqual(row["y_ret_5d"], nxt["close"] / row["close"] - 1, places=10)


if __name__ == "__main__":
    unittest.main()
