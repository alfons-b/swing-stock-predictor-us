"""Inti model challenger yang TIDAK butuh torch/statsmodels/arch (jalan di semua lingkungan).

- Rekursi ARIMA-GARCH dibuktikan dengan simulasi Monte Carlo (bukan dengan membandingkan ke library).
- Konversi kuantil → probabilitas, data panel tanpa look-ahead, split dengan embargo, metrik sama dengan model aktif.
"""
import unittest

import numpy as np
import pandas as pd

from app.challengers import arima_garch as ag
from app.challengers import data as cd
from app.challengers import dist
from app.challengers.evaluate import evaluate_forecasts, to_eval_frame, verdict


def simulate_arma_garch(n, mu, ar, ma, omega, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    r, eps, s2 = np.zeros(n), np.zeros(n), np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t:
            s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
        w = sum(ar[i] * (r[t - 1 - i] - mu) for i in range(len(ar)) if t - 1 - i >= 0)
        w += sum(ma[j] * eps[t - 1 - j] for j in range(len(ma)) if t - 1 - j >= 0)
        r[t] = mu + w + eps[t]
    return r, eps, s2


class TestArimaGarchRecursions(unittest.TestCase):
    P = {"mu": 0.05, "ar": [0.4], "ma": [-0.2], "omega": 0.05, "alpha": 0.08, "beta": 0.9}

    def test_residuals_and_variance_recover_truth(self):
        p = self.P
        r, eps, s2 = simulate_arma_garch(3000, p["mu"], p["ar"], p["ma"], p["omega"], p["alpha"], p["beta"])
        e_hat = ag.arma_residuals(r, p["mu"], p["ar"], p["ma"])
        np.testing.assert_allclose(e_hat[50:], eps[50:], atol=1e-8)
        s2_hat = ag.garch_variance(e_hat, p["omega"], p["alpha"], p["beta"], p["omega"] / (1 - p["alpha"] - p["beta"]))
        np.testing.assert_allclose(s2_hat[200:], s2[200:], rtol=1e-6)

    def test_h_step_mean_and_variance_match_monte_carlo(self):
        p, h = self.P, 5
        r, eps, s2 = simulate_arma_garch(800, p["mu"], p["ar"], p["ma"], p["omega"], p["alpha"], p["beta"], seed=1)
        params = {**p, "s0": p["omega"] / (1 - p["alpha"] - p["beta"])}
        mu_h, sd_h = ag.forecast_series(r / ag.SCALE, params, h)     # fungsi menerima log return satuan asli
        t = len(r) - 1
        rng = np.random.default_rng(7)
        n_sims = 40000
        sums = np.zeros(n_sims)
        prev_r = np.repeat(r[t], n_sims)
        prev_e = np.repeat(eps[t], n_sims)
        prev_s2 = np.repeat(s2[t], n_sims)
        for _ in range(h):
            s2n = p["omega"] + p["alpha"] * prev_e ** 2 + p["beta"] * prev_s2
            en = np.sqrt(s2n) * rng.standard_normal(n_sims)
            rn = p["mu"] + p["ar"][0] * (prev_r - p["mu"]) + p["ma"][0] * prev_e + en
            sums += rn
            prev_r, prev_e, prev_s2 = rn, en, s2n
        self.assertAlmostEqual(mu_h[t] * ag.SCALE, sums.mean(), delta=4 * sums.std() / np.sqrt(n_sims))
        self.assertAlmostEqual(sd_h[t] * ag.SCALE / sums.std(), 1.0, delta=0.03)
        # rata-rata: rumus analitik ARMA(1,1) dari state (r_t, ε_t) — deterministik, toleransi ketat
        for tt in (t, t - 37, t - 200):
            e1 = p["mu"] + p["ar"][0] * (r[tt] - p["mu"]) + p["ma"][0] * eps[tt]
            exp_sum, prev = 0.0, e1
            for k in range(h):
                exp_sum += prev
                prev = p["mu"] + p["ar"][0] * (prev - p["mu"])
            self.assertAlmostEqual(mu_h[tt] * ag.SCALE, exp_sum, places=8)

    def test_no_lookahead(self):
        p = {**self.P, "s0": 1.0}
        r, *_ = simulate_arma_garch(400, p["mu"], p["ar"], p["ma"], p["omega"], p["alpha"], p["beta"], seed=3)
        a = ag.forecast_series(r / 100, p, 5)
        r2 = r.copy()
        r2[300:] += 50                                                  # ubah masa depan setelah t = 299
        b = ag.forecast_series(r2 / 100, p, 5)
        np.testing.assert_allclose(a[0][:300], b[0][:300])
        np.testing.assert_allclose(a[1][:300], b[1][:300])

    def test_ewma_fallback_and_artifact_roundtrip(self):
        p = {"mu": 0.0, "ar": [], "ma": [], "s0": 1.0, "fallback": "ewma", "lambda": 0.94}
        mu, sd = ag.forecast_series(np.random.default_rng(0).normal(0, 0.01, 300), p, 5)
        self.assertTrue(np.isfinite(mu).all() and (sd > 0).all())
        blob = ag.to_artifact({"AAPL": p}, {"h": 5})
        params, meta = ag.from_artifact(blob)
        self.assertEqual(params["AAPL"]["fallback"], "ewma")
        self.assertEqual(meta["h"], 5)

    def test_psi_weights(self):
        np.testing.assert_allclose(ag.psi_weights([0.5], [], 4), [1, 0.5, 0.25, 0.125])
        np.testing.assert_allclose(ag.psi_weights([], [0.3], 3), [1, 0.3, 0.0])


class TestQuantileDistribution(unittest.TestCase):
    LV = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    def test_normal_quantiles_round_trip_probabilities(self):
        mu, sig = np.array([0.01, -0.02]), np.array([0.05, 0.03])
        q = dist.normal_quantiles(mu, sig, self.LV)
        from scipy.stats import norm
        for x in (-0.03, 0.0, 0.029):
            np.testing.assert_allclose(dist.quantile_cdf(q, self.LV, x), norm.cdf((x - mu) / sig), atol=0.012)
        np.testing.assert_allclose(dist.sigma_from_quantiles(q, self.LV), sig, rtol=1e-6)

    def test_summary_is_consistent(self):
        q = dist.normal_quantiles(np.array([0.0, 0.05]), np.array([0.04, 0.04]), self.LV)
        s = dist.summarize_quantiles(q, self.LV, 0.03, -0.03)
        self.assertTrue(np.all((s["prob_bullish"] + s["prob_bearish"] + s["prob_neutral"]) <= 1 + 1e-9))
        self.assertGreater(s["prob_up"][1], s["prob_up"][0])
        self.assertGreater(s["prob_bullish"][1], s["prob_bullish"][0])
        self.assertAlmostEqual(float(s["prob_up"][0]), 0.5, places=6)
        self.assertTrue(np.all(s["q10"] < s["q50"]) and np.all(s["q50"] < s["q90"]))

    def test_crossing_quantiles_are_sorted(self):
        q = np.array([[0.05, -0.05, 0.0, 0.01, 0.02, 0.03, 0.04]])     # tidak monoton
        s = dist.summarize_quantiles(q, self.LV, 0.03, -0.03)
        self.assertTrue(0 <= s["prob_up"][0] <= 1)


def _panel_df(n_days=40, tickers=("AAAA", "BBBB")):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    rng = np.random.default_rng(0)
    for t in tickers:
        px = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "close": px[i], "volume": 1e6 + i, "f_a": float(i), "f_b": np.nan,
                         "sector": "Energy", "is_tradeable": True,
                         "y_ret_5d": px[i + 5] / px[i] - 1 if i + 5 < n_days else np.nan})
    df = pd.DataFrame(rows)
    df["y_class"] = np.where(df["y_ret_5d"] > 0.03, 2, np.where(df["y_ret_5d"] < -0.03, 0, 1)).astype(float)
    df.loc[df["y_ret_5d"].isna(), "y_class"] = np.nan
    return df


class TestPanel(unittest.TestCase):
    def test_windows_stay_inside_one_stock_and_in_the_past(self):
        df = _panel_df()
        p = cd.prepare_panel(df, ["f_a"] + cd.ENGINEERED, 5)
        ends = cd.eligible_ends(p, 10)
        idx = cd.window_index(ends, 10)
        self.assertTrue((p.stock[idx] == p.stock[ends][:, None]).all())
        self.assertTrue((p.dates[idx] <= p.dates[ends][:, None]).all())
        self.assertTrue((p.pos[ends] >= 9).all())
        self.assertTrue(np.isfinite(p.y[ends]).all())                  # butuh target
        # target multi-horizon konsisten dengan label asli y_ret_5d (close yang sama)
        np.testing.assert_allclose(p.y_multi[ends, -1], np.log1p(p.y_ret[ends]), rtol=1e-5)
        lat = cd.latest_ends(p, 10)
        self.assertEqual(len(lat), 2)
        self.assertTrue((p.dates[lat] == p.dates.max()).all())

    def test_feature_selection_uses_coverage_only(self):
        df = _panel_df()
        f = cd.select_sequence_features(df, np.ones(len(df), bool), max_features=10, min_coverage=0.5)
        self.assertIn("f_a", f)
        self.assertNotIn("f_b", f)                                     # 0% cakupan
        self.assertEqual(f[-3:], cd.ENGINEERED)

    def test_scaler_from_train_rows_only(self):
        X = np.array([[1.0], [2.0], [3.0], [1000.0]], dtype=np.float32)
        c, s = cd.fit_scaler(X, np.array([0, 1, 2]))
        self.assertAlmostEqual(float(c[0]), 2.0)
        Z = cd.apply_scaler(np.array([[np.nan], [1000.0]], np.float32), c, s, clip=5)
        self.assertEqual(Z[0, 0], 0.0)
        self.assertEqual(Z[1, 0], 5.0)

    def test_future_dows_skip_weekends(self):
        fri = np.array(["2026-10-09"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(cd.future_dows(fri, 5)[0], [0, 1, 2, 3, 4])


class TestEvaluation(unittest.TestCase):
    def test_metrics_and_verdict(self):
        from tests.helpers import make_cfg
        cfg = make_cfg()
        rng = np.random.default_rng(0)
        n_days, n_names = 60, 40
        dates = np.repeat(pd.bdate_range("2026-01-01", periods=n_days).to_numpy(), n_names)
        signal = rng.normal(0, 0.02, len(dates))
        y_ret = signal + rng.normal(0, 0.03, len(dates))
        panel = cd.Panel(tickers=np.tile([f"T{i:02d}" for i in range(n_names)], n_days), dates=dates,
                         stock=np.zeros(len(dates), int), pos=np.zeros(len(dates), int), X=np.zeros((len(dates), 1), np.float32),
                         y=np.log1p(y_ret).astype(np.float32), y_multi=np.zeros((len(dates), 5), np.float32), y_ret=y_ret,
                         y_class=np.where(y_ret > 0.03, 2, np.where(y_ret < -0.03, 0, 1)).astype(float),
                         tradeable=np.ones(len(dates), bool), sector=np.zeros(len(dates), int),
                         dow=np.zeros(len(dates), int), close=np.ones(len(dates)))
        q = dist.normal_quantiles(np.log1p(signal), np.full(len(dates), 0.03), [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        fc = dist.summarize_quantiles(q, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], 0.03, -0.03)
        ev = to_eval_frame(panel, np.arange(len(dates)), fc)
        m = evaluate_forecasts(ev, cfg)
        self.assertEqual(m["status"], "OK")
        self.assertGreater(m["roc_auc_bullish"], 0.6)                  # sinyal nyata → AUC tinggi
        self.assertGreater(m["ic_mean"], 0.2)
        self.assertAlmostEqual(m["coverage_80"], 0.8, delta=0.05)       # σ benar → cakupan ≈ nominal
        for key in ("log_loss", "topk_mean_net_return", "ece_bullish", "auc_up", "pinball_mean"):
            self.assertIsNotNone(m[key])
        msgs = verdict(m, {"momentum": {"topk_mean_net_return": -0.01}}, {"version": "model_v001", "topk_mean_net_return": None,
                                                                           "roc_auc_bullish": None})
        self.assertTrue(any("momentum" in s for s in msgs))


if __name__ == "__main__":
    unittest.main()


class TestRegistryAndReport(unittest.TestCase):
    """Registry & prakiraan di database (SQLite; jalur PostgreSQL di CI) + laporan pembanding + evaluasi live."""

    def setUp(self):
        from tests.helpers import make_cfg, make_ctx
        self.ctx = make_ctx(make_cfg())
        self.ctx.repo.upsert_stocks(pd.DataFrame({"ticker": ["AAAA", "BBBB"], "is_active": True}))

    def tearDown(self):
        self.ctx.close()

    def test_registry_versions_artifacts_and_report(self):
        from app.challengers.report import comparison_table
        repo = self.ctx.repo
        self.assertEqual(repo.next_challenger_version("lstm"), "lstm_v001")
        for i in range(3):
            repo.save_challenger_model({"model_name": "lstm", "version": f"lstm_v00{i + 1}", "status": "READY",
                                        "created_at": f"2026-10-1{i}T00:00:00+00:00", "artifact": b"abc",
                                        "metrics": {"test": {"roc_auc_bullish": 0.55 + i / 100, "topk_mean_net_return": 0.001,
                                                             "eval_start": "2026-04-10", "eval_end": "2026-10-09", "n": 100}},
                                        "params": {"features": ["f_a"]}})
        self.assertEqual(repo.next_challenger_version("lstm"), "lstm_v004")
        self.assertEqual(repo.prune_challenger_artifacts("lstm", 2), 1)
        self.assertEqual(repo.latest_challenger("lstm")["version"], "lstm_v003")
        reg = repo.challenger_models("lstm")
        self.assertEqual(reg.loc[reg["version"] == "lstm_v001", "status"].iloc[0], "ARCHIVED")
        t = comparison_table(repo)
        self.assertEqual(t.iloc[0]["versi"], "lstm_v003")
        self.assertAlmostEqual(t.iloc[0]["AUC bullish"], 0.57)
        # model aktif tidak tersentuh
        self.assertEqual(int(self.ctx.db.scalar("SELECT COUNT(*) FROM model_versions")), 0)

    def test_forecasts_upsert_and_live_evaluation(self):
        from app.challengers.report import live_evaluation
        repo = self.ctx.repo
        f = pd.DataFrame({"forecast_date": "2026-10-01", "ticker": ["AAAA", "BBBB", "ZZZZ"], "model_name": "gru",
                          "model_version": "gru_v001", "horizon": 5, "exp_return": [0.01, -0.01, 0.0],
                          "prob_up": 0.5, "prob_bullish": [0.4, 0.1, 0.2], "prob_bearish": 0.1, "vol": 0.05,
                          "q10": -0.05, "q50": 0.0, "q90": 0.05})
        repo.save_model_forecasts(f)
        repo.save_model_forecasts(f)                                     # idempoten
        got = repo.model_forecasts(model_name="gru")
        self.assertEqual(len(got), 2)                                    # ZZZZ tidak ada di universe → dilewati
        dates = pd.bdate_range("2026-10-01", periods=7)
        prices = pd.DataFrame({"ticker": np.repeat(["AAAA", "BBBB"], 7), "date": np.tile(dates, 2),
                               "close": np.r_[np.linspace(100, 110, 7), np.linspace(100, 90, 7)]})
        live = live_evaluation(got, prices, 5, 0.03)
        self.assertEqual(live.iloc[0]["n"], 2)
        self.assertEqual(live.iloc[0]["AUC bullish"], 1.0)


class TestDatasetNpz(unittest.TestCase):
    def test_roundtrip_without_pickle(self):
        import tempfile
        from app.challengers.runner import load_dataset_npz, save_dataset_npz
        df = _panel_df(10)
        df["is_tradeable"] = [True, False] * 10
        path = tempfile.mktemp(suffix=".npz")
        save_dataset_npz(df[["date", "ticker", "sector", "close", "volume", "is_tradeable", "f_a", "y_ret_5d", "y_class"]], path)
        back = load_dataset_npz(path)
        self.assertEqual(len(back), len(df))
        self.assertEqual(back["is_tradeable"].dtype, bool)
        self.assertEqual(back["is_tradeable"].tolist(), df["is_tradeable"].tolist())
        np.testing.assert_allclose(back["close"], df["close"], rtol=1e-6)
        self.assertEqual(back["date"].iloc[3], df["date"].iloc[3])


import os  # noqa: E402


@unittest.skipUnless(os.environ.get("TEST_POSTGRES_URL"), "butuh TEST_POSTGRES_URL")
class TestRegistryAndReportPostgres(TestRegistryAndReport):
    """Registry & prakiraan di PostgreSQL sungguhan (artifact BYTEA, JSON TEXT, DATE)."""

    def setUp(self):
        from app.database.db import Database
        from app.database.schema import TABLE_NAMES
        from tests.helpers import make_cfg, make_ctx
        db = Database.from_url(os.environ["TEST_POSTGRES_URL"])
        db.executescript([f"DROP TABLE IF EXISTS {t} CASCADE" for t in reversed(TABLE_NAMES)])
        db.close()
        self.ctx = make_ctx(make_cfg(db_url=os.environ["TEST_POSTGRES_URL"]))
        self.ctx.repo.upsert_stocks(pd.DataFrame({"ticker": ["AAAA", "BBBB"], "is_active": True}))

    def test_binary_artifact_roundtrip(self):
        from app.challengers import arima_garch as ag
        blob = ag.to_artifact({"AAAA": {"status": "OK", "mu": 0.1, "ar": [0.2], "ma": []}}, {"horizon": 5})
        self.ctx.repo.save_challenger_model({"model_name": "arima_garch", "version": "arima_garch_v001", "status": "READY",
                                             "created_at": "2026-10-15T00:00:00+00:00", "artifact": blob})
        got = self.ctx.repo.latest_challenger("arima_garch")
        params, meta = ag.from_artifact(got["artifact"])
        self.assertEqual(params["AAAA"]["ar"], [0.2])


class TestSignificanceCalibrationRanking(unittest.TestCase):
    """Uji signifikansi (bootstrap blok), kalibrasi Platt, dan pemilihan aturan ranking di fold validasi."""

    def test_block_bootstrap_null_and_signal(self):
        from app.challengers.evaluate import block_bootstrap
        rng = np.random.default_rng(1)
        null = block_bootstrap(rng.normal(0, 0.02, 120), block=5, n_boot=2000, seed=0)
        self.assertLessEqual(null["ci_low"], 0)
        self.assertGreaterEqual(null["ci_high"], 0)
        self.assertGreater(null["p_value"], 0.05)
        sig = block_bootstrap(rng.normal(0.01, 0.02, 120), block=5, n_boot=2000, seed=0)
        self.assertGreater(sig["ci_low"], 0)
        self.assertLess(sig["p_value"], 0.01)
        self.assertEqual(block_bootstrap([0.1, 0.2], 5)["ci_low"], None)        # terlalu pendek → tidak dinilai
        x = rng.normal(0, 1, 100)
        self.assertEqual(block_bootstrap(x, 5, 500, seed=3), block_bootstrap(x, 5, 500, seed=3))   # deterministik

    def test_block_bootstrap_is_wider_for_overlapping_returns(self):
        """Return 5 hari dari tanggal berurutan saling bertumpuk → CI blok harus lebih lebar dari CI per tanggal."""
        from app.challengers.evaluate import block_bootstrap
        rng = np.random.default_rng(2)
        daily = rng.normal(0, 0.02, 600)
        overlapping = np.convolve(daily, np.ones(5), mode="valid")[:400]
        w_block = block_bootstrap(overlapping, 5, 2000, 0)
        w_iid = block_bootstrap(overlapping, 1, 2000, 0)
        self.assertGreater(w_block["ci_high"] - w_block["ci_low"], 1.3 * (w_iid["ci_high"] - w_iid["ci_low"]))

    def _ev(self, n_days=120, n_names=40, seed=0):
        rng = np.random.default_rng(seed)
        dates = np.repeat(pd.bdate_range("2026-01-01", periods=n_days), n_names)
        vol = np.tile(np.linspace(0.02, 0.15, n_names), n_days)            # saham makin volatil
        alpha = rng.normal(0, 0.01, len(dates))
        y = alpha - 0.4 * vol + rng.normal(0, 1, len(dates)) * vol         # yang volatil justru jatuh (pasar turun)
        from scipy.stats import norm
        mu = alpha - 0.4 * vol
        ev = pd.DataFrame({"date": dates, "ticker": np.tile([f"T{i}" for i in range(n_names)], n_days),
                           "is_tradeable": True, "y_ret": y,
                           "y_class": np.where(y > 0.03, 2, np.where(y < -0.03, 0, 1)).astype(float),
                           "exp_return": mu, "vol": vol, "prob_up": 1 - norm.cdf(-mu / vol),
                           "prob_bullish": 1 - norm.cdf((0.03 - mu) / vol), "prob_bearish": norm.cdf((-0.03 - mu) / vol)})
        ev["prob_neutral"] = 1 - ev["prob_bullish"] - ev["prob_bearish"]
        return ev

    def test_ranking_rule_avoids_volatility_bias(self):
        from app.challengers.evaluate import choose_ranking
        from tests.helpers import make_cfg
        rule, scores = choose_ranking(self._ev(), make_cfg())
        self.assertNotEqual(rule, "prob_bullish")                          # P(>3%) memilih saham paling volatil
        self.assertGreater(scores[rule], scores["prob_bullish"])
        self.assertEqual(set(scores), {"prob_bullish", "exp_return", "risk_adjusted", "prob_up"})

    def test_paired_significance_vs_baselines(self):
        from app.challengers.evaluate import significance
        from tests.helpers import make_cfg
        ev = self._ev()
        ev["rank"] = ev["exp_return"]
        noise = np.random.default_rng(9).normal(size=len(ev))               # baseline tanpa informasi
        s = significance(ev, make_cfg(), "rank", noise)
        self.assertGreater(s["topk_vs_momentum"]["ci_low"], 0)
        self.assertIn("p_value", s["ic"])

    def test_platt_fixes_overconfidence_without_changing_order(self):
        from sklearn.metrics import roc_auc_score
        from app.challengers.calibration import apply_platt, calibrate_forecasts, fit_calibrators
        from app.models.evaluation import expected_calibration_error
        rng = np.random.default_rng(4)
        z = rng.normal(-1.0, 1.0, 20000)
        y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-z))).astype(int)
        p_over = 1 / (1 + np.exp(-2.5 * z))                                # terlalu yakin
        fc = {"prob_bullish": p_over, "prob_bearish": np.full(z.size, 0.2), "prob_up": p_over}
        y_ret = np.where(y == 1, 0.05, 0.0)
        cal = fit_calibrators(fc, y_ret, 0.03, -0.03)
        self.assertIn("prob_bullish", cal)
        p_cal = apply_platt(p_over, cal["prob_bullish"])
        self.assertLess(expected_calibration_error(y, p_cal), 0.5 * expected_calibration_error(y, p_over))
        self.assertAlmostEqual(roc_auc_score(y, p_cal), roc_auc_score(y, p_over), places=5)   # monoton (seri di batas klip)
        out = calibrate_forecasts({**fc, "prob_bearish": np.full(z.size, 0.9)}, cal)
        self.assertTrue(np.all(out["prob_bullish"] + out["prob_bearish"] <= 1.0 + 1e-9))
        self.assertTrue(np.all(out["prob_neutral"] >= 0))

    def test_verdict_uses_significance_words(self):
        from app.challengers.evaluate import verdict
        m = {"status": "OK", "roc_auc_bullish": 0.6, "auc_up": 0.55, "ece_bullish": 0.05, "ece_bullish_raw": 0.1,
             "ranking_rule": "risk_adjusted", "topk_by_rule_net": 0.004, "topk_mean_net_return": 0.002}
        sig = {"topk": {"mean": 0.004, "ci_low": -0.003, "ci_high": 0.011, "p_value": 0.3},
               "topk_vs_reversal": {"mean": 0.02, "ci_low": 0.005, "ci_high": 0.03, "p_value": 0.01},
               "topk_vs_momentum": {"mean": 0.001, "ci_low": -0.01, "ci_high": 0.012, "p_value": 0.8}}
        txt = " | ".join(verdict(m, {"reversal": {"topk_mean_net_return": -0.02}, "momentum": {"topk_mean_net_return": 0.0}},
                                 {"version": "model_v001", "topk_mean_net_return": 0.011, "roc_auc_bullish": 0.58,
                                  "comparison": "di dalam CI 95% challenger — perbedaan tidak signifikan"}, sig))
        self.assertIn("lebih baik secara SIGNIFIKAN", txt)
        self.assertIn("perbedaan TIDAK signifikan", txt)
        self.assertIn("lolos syarat", txt)
        self.assertIn("TIDAK terbukti > 0", txt)
        self.assertIn("risk_adjusted", txt)


class TestSmallUniverse(unittest.TestCase):
    def test_daily_ic_and_significance_with_few_names(self):
        """< 20 emiten per tanggal (data contoh): IC tidak dinilai, tetapi tidak boleh error."""
        from app.challengers.evaluate import block_bootstrap, daily_ic, significance
        from tests.helpers import make_cfg
        ev = TestSignificanceCalibrationRanking()._ev(n_days=60, n_names=12)
        ic = daily_ic(ev, "exp_return", "y_ret")
        self.assertIsInstance(ic, pd.Series)
        self.assertTrue(ic.empty)
        self.assertIsNone(block_bootstrap(ic)["ci_low"])
        s = significance(ev, make_cfg(), "exp_return", ev["exp_return"].to_numpy())
        self.assertIsNone(s["ic"]["ci_low"])
        self.assertIsNotNone(s["topk"]["ci_low"])

    def test_verdict_never_fails_on_missing_values(self):
        from app.challengers.evaluate import verdict
        m = {"status": "OK", "roc_auc_bullish": 0.55, "auc_up": None, "ece_bullish": None, "ece_bullish_raw": None,
             "ranking_rule": "exp_return", "topk_by_rule_net": None, "topk_mean_net_return": -0.01}
        sig = {"topk": {"mean": None, "ci_low": None, "ci_high": None, "p_value": None},
               "ic": {"mean": None, "ci_low": None, "ci_high": None, "p_value": None},
               "topk_vs_momentum": {"mean": 0.001, "ci_low": -0.01, "ci_high": 0.01, "p_value": None}}
        out = verdict(m, {"momentum": {"topk_mean_net_return": None}, "reversal": {"topk_mean_net_return": -0.02}},
                      {"version": "v", "topk_mean_net_return": 0.01, "roc_auc_bullish": None}, sig)
        self.assertTrue(out)
        self.assertTrue(any("n/a" in x for x in out))
