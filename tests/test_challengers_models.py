"""Model challenger dengan library sungguhan (torch, statsmodels, arch). Dilewati bila library tidak terpasang;
dijalankan oleh workflow `challenger_tests.yml` (CI GitHub, CPU)."""
import importlib.util
import unittest

import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TS = importlib.util.find_spec("statsmodels") is not None and importlib.util.find_spec("arch") is not None
LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _tiny_panel(n_stocks=6, n_days=160, n_feat=5, seed=0):
    from app.challengers.data import Panel
    rng = np.random.default_rng(seed)
    N = n_stocks * n_days
    stock = np.repeat(np.arange(n_stocks), n_days)
    pos = np.tile(np.arange(n_days), n_stocks)
    dates = np.tile(pd.bdate_range("2024-01-01", periods=n_days).to_numpy(), n_stocks)
    X = rng.normal(size=(N, n_feat)).astype(np.float32)
    y = (0.6 * X[:, 0] + 0.3 * rng.normal(size=N)).astype(np.float32) * 0.03       # sinyal di fitur 0 hari t
    y_multi = np.cumsum(np.repeat((y / 5)[:, None], 5, axis=1), axis=1).astype(np.float32)
    return Panel(tickers=np.repeat([f"S{i}" for i in range(n_stocks)], n_days), dates=dates, stock=stock, pos=pos, X=X,
                 y=y, y_multi=y_multi, y_ret=np.expm1(y), y_class=np.ones(N), tradeable=np.ones(N, bool),
                 sector=np.repeat(np.arange(n_stocks) % 3 + 1, n_days), dow=pd.to_datetime(dates).dayofweek.to_numpy(),
                 close=np.ones(N), features=[f"f{i}" for i in range(n_feat)], sectors=["A", "B", "C"])


def _spec(kind, panel, seq_len=10, arch=None):
    from app.challengers.trainer import SequenceSpec
    t = panel.y_multi if kind == "tft" else panel.y
    return SequenceSpec(kind, panel.features, panel.sectors, seq_len, 5, LEVELS,
                        arch or ({"hidden": 8, "heads": 2, "dropout": 0.0} if kind == "tft" else {"hidden": 8, "layers": 1, "dropout": 0.0}),
                        np.zeros(panel.X.shape[1]), np.ones(panel.X.shape[1]), t.mean(axis=0), t.std(axis=0) + 1e-6, 5.0)


@unittest.skipUnless(HAS_TORCH, "butuh torch")
class TestNets(unittest.TestCase):
    def test_shapes_and_monotone_quantiles(self):
        import torch
        from app.challengers.trainer import _batch
        p = _tiny_panel()
        ends = np.array([20, 50, 300, 700])
        for kind in ("lstm", "gru", "tft"):
            spec = _spec(kind, p)
            m = spec.new_model().eval()
            with torch.no_grad():
                out = m(*_batch(spec, p, p.X, ends))
            self.assertEqual(tuple(out.shape), (4, 5, 7) if kind == "tft" else (4, 7), kind)
            self.assertTrue(bool((out.diff(dim=-1) >= 0).all()), f"{kind}: kuantil bersilangan")

    def test_no_lookahead_in_windows(self):
        from app.challengers.trainer import predict_quantiles
        p = _tiny_panel()
        ends = np.array([40, 41, 200])
        for kind in ("lstm", "gru", "tft"):
            spec = _spec(kind, p)
            m = spec.new_model()
            a = predict_quantiles(m, spec, p, p.X, ends)
            Z2 = p.X.copy()
            Z2[42:160] += 100.0                                         # ubah masa depan emiten 0 setelah baris 41
            Z2[201:320] -= 100.0
            b = predict_quantiles(m, spec, p, Z2, ends)
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg=kind)

    def test_tft_attention_is_causal_and_weights_sum_to_one(self):
        import torch
        from app.challengers.trainer import _batch
        p = _tiny_panel()
        spec = _spec("tft", p)
        m = spec.new_model().eval()
        with torch.no_grad():
            _, w = m(*_batch(spec, p, p.X, np.array([100, 101])), return_weights=True)
        att = w["attention"]                                            # (B, H, L+H)
        L = spec.seq_len
        for i in range(spec.horizon):
            self.assertLess(float(att[:, i, L + i + 1:].abs().sum()), 1e-6)
        self.assertAlmostEqual(float(w["past_variables"].sum()), 1.0, places=4)

    def test_artifact_roundtrip_weights_only(self):
        from app.challengers.trainer import load_artifact, predict_quantiles, save_artifact
        p = _tiny_panel()
        ends = np.array([30, 400])
        for kind in ("lstm", "gru", "tft"):
            spec = _spec(kind, p)
            m = spec.new_model()
            blob = save_artifact(m, spec)
            m2, spec2 = load_artifact(blob)
            np.testing.assert_allclose(predict_quantiles(m, spec, p, p.X, ends),
                                       predict_quantiles(m2, spec2, p, p.X, ends), rtol=1e-6)
            self.assertEqual(spec2.features, spec.features)

    def test_training_learns_signal(self):
        from app.challengers.trainer import fit_sequence_model, predict_quantiles
        p = _tiny_panel(n_stocks=8, n_days=220)
        train = np.flatnonzero((p.pos >= 9) & (p.pos < 160))
        val = np.flatnonzero(p.pos >= 170)
        tr = {"batch_size": 128, "samples_per_epoch": 1000, "max_epochs": 25, "min_epochs": 3, "patience": 5,
              "learning_rate": 0.01, "max_train_minutes": 3}
        for kind in ("lstm", "gru", "tft"):
            spec = _spec(kind, p)
            model, info = fit_sequence_model(spec, p, p.X, train, tr, seed=0, val_ends=val)
            q = predict_quantiles(model, spec, p, p.X, val)
            med = q[:, 3]
            corr = np.corrcoef(med, p.y[val])[0, 1]
            self.assertGreater(corr, 0.5, f"{kind}: korelasi median vs target {corr:.2f}; {info['history'][-3:]}")
            self.assertGreaterEqual(info["best_epoch"], 1)


@unittest.skipUnless(HAS_TS, "butuh statsmodels & arch")
class TestArimaGarchEstimation(unittest.TestCase):
    def test_recovers_parameters_and_matches_libraries(self):
        from tests.test_challengers_core import simulate_arma_garch
        from app.challengers import arima_garch as ag
        r, eps, s2 = simulate_arma_garch(2500, 0.05, [0.4], [], 0.05, 0.08, 0.9, seed=11)
        p = ag.fit_one(r / ag.SCALE, [(0, 0), (1, 0), (0, 1), (1, 1)], {"p": 1, "q": 1, "dist": "t"})
        self.assertEqual(p["status"], "OK")
        self.assertNotIn("fallback", p)
        self.assertIn(p["p"], (1, 2))
        if p["p"] == 1 and p["q"] == 0:
            self.assertAlmostEqual(p["ar"][0], 0.4, delta=0.08)
        self.assertAlmostEqual(p["alpha"] + p["beta"], 0.98, delta=0.05)
        # rata-rata 5 hari: rekursi kita == statsmodels.forecast (parameter sama, akhir sampel)
        from statsmodels.tsa.arima.model import ARIMA
        res = ARIMA(r, order=(p["p"], 0, p["q"]), trend="c").fit()
        mu_h, sd_h = ag.forecast_series(r / ag.SCALE, p, 5)
        self.assertAlmostEqual(mu_h[-1] * ag.SCALE, float(np.sum(res.forecast(5))), delta=0.02)
        # varians 1..5 hari: rekursi kita == arch.forecast pada residual yang sama (ARMA → GARCH dua tahap)
        from arch import arch_model
        e = ag.arma_residuals(r, p["mu"], p["ar"], p["ma"])
        g = arch_model(e, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        fixed = g.fix([p["omega"], p["alpha"], p["beta"], p["nu"]])
        lib = fixed.forecast(horizon=5, reindex=False).variance.to_numpy()[-1]
        s2_ours = ag.garch_variance(e, p["omega"], p["alpha"], p["beta"], p["s0"])
        ours = ag.garch_forecast_paths(e, s2_ours, p["omega"], p["alpha"], p["beta"], 5)[-1]
        np.testing.assert_allclose(ours, lib, rtol=0.02)

    def test_insufficient_data_and_parallel(self):
        from app.challengers import arima_garch as ag
        rng = np.random.default_rng(0)
        out = ag.fit_all({"A": rng.normal(0, 0.01, 400), "B": rng.normal(0, 0.01, 50)}, [(0, 0), (1, 0)],
                         {"p": 1, "q": 1, "dist": "t"}, 250, n_jobs=2)
        self.assertEqual(out["A"]["status"], "OK")
        self.assertEqual(out["B"]["status"], "INSUFFICIENT_DATA")


@unittest.skipUnless(HAS_TORCH and HAS_TS, "butuh torch, statsmodels & arch")
class TestEndToEnd(unittest.TestCase):
    """Data contoh (SQLite / PostgreSQL CI): setup model aktif → latih 4 challenger → prakiraan → model aktif utuh."""

    @classmethod
    def setUpClass(cls):
        from app.pipeline.jobs import job_setup
        from tests.helpers import make_cfg, make_ctx
        cls.cfg = make_cfg(as_of="2020-12-31")
        cls.cfg["challengers"]["sequence"].update({"seq_len": 20, "max_features": 12})
        cls.cfg["challengers"]["training"].update({"batch_size": 128, "samples_per_epoch": 1500, "max_epochs": 2,
                                                   "min_epochs": 1, "patience": 1, "max_train_minutes": 2,
                                                   "val_samples": 2000})
        cls.cfg["challengers"]["lstm"] = {"hidden": 8, "layers": 1, "dropout": 0.0}
        cls.cfg["challengers"]["gru"] = {"hidden": 8, "layers": 1, "dropout": 0.0}
        cls.cfg["challengers"]["tft"] = {"hidden": 8, "heads": 2, "dropout": 0.0, "seq_len": 15}
        cls.cfg["challengers"]["arima_garch"].update({"window": 500, "n_jobs": 2, "orders": [[0, 0], [1, 0]]})
        cls.cfg["challengers"]["min_history_days"] = 120
        job_setup(make_ctx(cls.cfg), skip_backtest=True)
        cls.ctx = make_ctx(cls.cfg)
        cls.active_before = cls.ctx.db.query_df("SELECT version, status, artifact FROM model_versions ORDER BY version")
        from app.challengers.runner import prepare_dataset
        cls.df = prepare_dataset(cls.cfg, cls.ctx.repo)

    def test_train_all_predict_and_active_model_untouched(self):
        from app.challengers import MODEL_NAMES
        from app.challengers.runner import prepare_dataset, run_predict, run_train
        repo = self.ctx.repo
        for m in MODEL_NAMES:
            out = run_train(m, self.df, self.cfg, repo, f"test-{m}")
            self.assertIn(out["test"].get("status"), ("OK", "INSUFFICIENT_DATA"), (m, out["test"]))
            self.assertTrue(out["verdict"])
            self.assertEqual(repo.latest_challenger(m)["version"], out["version"])
            # lapisan validasi (kalibrasi + aturan ranking) tersimpan dan dipakai; signifikansi dilaporkan
            import json
            params = json.loads(repo.latest_challenger(m)["params"])
            self.assertIn(params["ranking_rule"], ("prob_bullish", "exp_return", "risk_adjusted", "prob_up"), m)
            self.assertIn("calibrators", params)
            if out["test"].get("status") == "OK":
                self.assertIn("topk", out["significance"], m)
                self.assertIn("ece_bullish_raw", out["test"])
                self.assertEqual(out["test"]["ranking_rule"], params["ranking_rule"])
        recent = prepare_dataset(self.cfg, repo, 200, labels=False)
        res = run_predict(list(MODEL_NAMES), recent, self.cfg, repo, "test-predict")
        for m in MODEL_NAMES:
            self.assertEqual(res[m]["status"], "OK", (m, res[m]))
            self.assertGreater(res[m]["n"], 0)
        f = repo.model_forecasts()
        self.assertEqual(set(f["model_name"]), set(MODEL_NAMES))
        self.assertTrue(((f["prob_bullish"] >= 0) & (f["prob_bullish"] <= 1)).all())
        self.assertTrue((f["q10"] <= f["q90"]).all())
        self.assertTrue(f["rank_score"].notna().all())
        after = self.ctx.db.query_df("SELECT version, status, artifact FROM model_versions ORDER BY version")
        pd.testing.assert_frame_equal(self.active_before, after)       # model aktif tidak berubah sedikit pun

    def test_no_train_serve_skew_in_features(self):
        """Fitur pada tanggal terakhir dari dataset prakiraan harian (lookback terbatas) == dari histori penuh."""
        from app.challengers import challenger_cfg
        from app.challengers import data as cd
        from app.challengers.runner import prepare_dataset
        lb = challenger_cfg(self.cfg)["predict_lookback_days"]
        recent = prepare_dataset(self.cfg, self.ctx.repo, lb, labels=False)
        feats = cd.select_sequence_features(self.df, np.ones(len(self.df), bool), 64, 0.6, self.cfg)
        full = cd.prepare_panel(self.df, feats, 5)
        part = cd.prepare_panel(recent, feats, 5)
        L = 60
        e_full, e_part = cd.latest_ends(full, L), cd.latest_ends(part, L)
        self.assertEqual(list(full.tickers[e_full]), list(part.tickers[e_part]))
        # Emiten yang harganya datar ≥ setahun (suspensi panjang) dikecualikan: RSI Wilder "membeku" pada nilai sebelum
        # suspensi (up & down meluruh bersama), dan nilai itu bergantung pada histori di luar jendela mana pun.
        # Emiten seperti ini berstatus cakupan STALE/EXCLUDED dan tidak pernah menjadi sinyal.
        r_idx = part.features.index("c_ret1")
        moved = np.array([np.nanmax(np.abs(part.X[max(0, e - 250):e + 1, r_idx])) > 0 for e in e_part])
        self.assertGreaterEqual(moved.mean(), 0.8)
        e_full, e_part = e_full[moved], e_part[moved]
        w_full, w_part = full.X[cd.window_index(e_full, L)], part.X[cd.window_index(e_part, L)]
        both = np.isfinite(w_full) & np.isfinite(w_part)
        self.assertTrue((np.isfinite(w_full) == np.isfinite(w_part)).all(), "NaN berbeda → lookback kurang panjang")
        np.testing.assert_allclose(w_part[both], w_full[both], rtol=1e-4, atol=1e-5)

    def test_daily_pipeline_still_works(self):
        from app.pipeline.jobs import job_daily
        from tests.helpers import make_ctx
        s = job_daily(make_ctx(self.cfg), skip_ingest=True)
        self.assertIn(s["run_status"], ("SUCCESS", "PARTIAL_SUCCESS"))


if __name__ == "__main__":
    unittest.main()
