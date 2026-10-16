"""Universe saham AS: file konstituen (Symbol | Security | GICS Sector | GICS Sub-Industry), tabel Wikipedia,
cadangan GitHub, dan pengaman deaktivasi saat sumber hanya parsial."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.data.universe import UniverseManager
from app.data.universe_file import UniverseFileError, apply_sectors, load_universe_file, parse_date, parse_shares
from app.data.us_universe import GITHUB_SP500, WIKI, USIndexUniverse, parse_wikipedia
from tests.helpers import make_cfg, make_ctx

SP_ROWS = [
    {"Symbol": "AAPL", "Security": "Apple Inc.", "GICS Sector": "Information Technology",
     "GICS Sub-Industry": "Technology Hardware, Storage & Peripherals", "Date added": "1982-11-30"},
    {"Symbol": "BRK.B", "Security": "Berkshire Hathaway", "GICS Sector": "Financials",
     "GICS Sub-Industry": "Multi-Sector Holdings", "Date added": "2010-02-16"},
    {"Symbol": "jpm ", "Security": "JPMorgan Chase", "GICS Sector": "Financials",
     "GICS Sub-Industry": "Diversified Banks", "Date added": "1975-06-30"},
]


def tmpdir():
    return Path(tempfile.mkdtemp(prefix="ssp_uni_"))


def wiki_html(rows: list[dict], extra_table: bool = True) -> str:
    """Halaman bergaya Wikipedia: tabel konstituen + tabel perubahan (lebih kecil) yang harus diabaikan."""
    t1 = pd.DataFrame(rows).to_html(index=False)
    t2 = pd.DataFrame({"Date": ["2026-09-22"], "Added": ["XYZ"], "Removed": ["OLD"]}).to_html(index=False) if extra_table else ""
    return f"<html><body><h2>Constituents</h2>{t1}<h2>Changes</h2>{t2}</body></html>"


class TestParsers(unittest.TestCase):
    def test_dates(self):
        for s in ("2000-05-31", "31/05/2000", "31-May-2000", 36677, "36677"):
            self.assertEqual(parse_date(s), pd.Timestamp("2000-05-31"), s)
        self.assertTrue(pd.isna(parse_date("-")))

    def test_shares(self):
        self.assertEqual(parse_shares("1,924,688,333"), 1924688333.0)
        self.assertEqual(parse_shares(5e9), 5e9)


class TestConstituentFile(unittest.TestCase):
    def _check(self, u):
        self.assertEqual(u["ticker"].tolist(), ["AAPL", "BRK-B", "JPM"])
        self.assertEqual(u.loc[1, "name"], "Berkshire Hathaway")
        self.assertEqual(u.loc[2, "sector"], "Financials")
        self.assertEqual(u.loc[0, "subsector"], "Technology Hardware, Storage & Peripherals")
        self.assertTrue(pd.isna(u.loc[0, "listing_date"]))         # "Date added" = masuk indeks, BUKAN tanggal IPO

    def test_xlsx_with_title_rows(self):
        p = tmpdir() / "constituents.xlsx"
        with pd.ExcelWriter(p) as w:  # ekspor kadang punya baris judul di atas tabel
            pd.DataFrame([["S&P 500 constituents"], [""]]).to_excel(w, index=False, header=False)
            pd.DataFrame(SP_ROWS).to_excel(w, index=False, startrow=3)
        self._check(load_universe_file(p))

    def test_csv_semicolon_and_comma(self):
        for sep in (",", ";"):
            p = tmpdir() / "constituents.csv"
            pd.DataFrame(SP_ROWS).to_csv(p, index=False, sep=sep)
            self._check(load_universe_file(p))

    def test_internal_format_still_works(self):
        p = tmpdir() / "u.csv"
        pd.DataFrame({"ticker": ["brk/b"], "sector": ["Financials"], "board": ["S&P 500"]}).to_csv(p, index=False)
        u = load_universe_file(p)
        self.assertEqual((u.loc[0, "ticker"], u.loc[0, "sector"], u.loc[0, "board"]), ("BRK-B", "Financials", "S&P 500"))

    def test_shipped_snapshot_is_valid(self):
        u = load_universe_file(Path(__file__).resolve().parents[1] / "config" / "universe.csv")
        self.assertGreater(len(u), 480)
        self.assertIn("BRK-B", set(u["ticker"]))
        self.assertTrue(u["sector"].notna().all())

    def test_missing_code_column(self):
        p = tmpdir() / "x.csv"
        pd.DataFrame({"Security": ["A"]}).to_csv(p, index=False)
        with self.assertRaises(UniverseFileError):
            load_universe_file(p)

    def test_sector_file_merge(self):
        d = tmpdir()
        pd.DataFrame({"Symbol": ["AAPL"], "Sector": ["Tech (custom)"], "Industry": ["Hardware"]}).to_csv(d / "s.csv", index=False)
        pd.DataFrame(SP_ROWS).drop(columns=["GICS Sector", "GICS Sub-Industry"]).to_csv(d / "u.csv", index=False)
        u = apply_sectors(load_universe_file(d / "u.csv"), d / "s.csv")
        self.assertEqual(u.set_index("ticker").loc["AAPL", "sector"], "Tech (custom)")
        self.assertTrue(pd.isna(u.set_index("ticker").loc["JPM", "sector"]))   # tidak ada di file sektor → kosong


class TestWikipediaUniverse(unittest.TestCase):
    def test_parse_picks_constituent_table(self):
        u = parse_wikipedia(wiki_html(SP_ROWS), "S&P 500", min_rows=3)
        self.assertEqual(u["ticker"].tolist(), ["AAPL", "BRK-B", "JPM"])
        self.assertEqual(set(u["board"]), {"S&P 500"})
        with self.assertRaises(ValueError):
            parse_wikipedia(wiki_html(SP_ROWS), "S&P 500", min_rows=400)       # tabel terpotong → gagal, bukan dipakai

    def _cfg(self):
        cfg = make_cfg()
        cfg["universe"]["indexes"] = ["sp500", "sp400"]
        return cfg

    def test_complete_when_all_indexes_ok(self):
        mid = [{**r, "Symbol": s} for r, s in zip(SP_ROWS, ("DECK", "AAPL", "WSM"))]   # AAPL ganda → sekali
        pages = {WIKI["sp500"][0]: wiki_html(SP_ROWS), WIKI["sp400"][0]: wiki_html(mid)}
        import app.data.us_universe as uu
        old = dict(uu.WIKI)
        uu.WIKI.update({k: (v[0], v[1], 2) for k, v in uu.WIKI.items()})
        try:
            u = USIndexUniverse(self._cfg(), http_get=lambda url: pages[url]).get_universe()
        finally:
            uu.WIKI.clear()
            uu.WIKI.update(old)
        self.assertTrue(u.attrs["complete"])
        self.assertEqual(sorted(u["ticker"]), ["AAPL", "BRK-B", "DECK", "JPM", "WSM"])
        self.assertEqual(u.set_index("ticker").loc["AAPL", "board"], "S&P 500")

    def test_partial_falls_back_to_github_and_is_marked_incomplete(self):
        csv = pd.DataFrame(SP_ROWS).to_csv(index=False)

        def get(url):
            if url == GITHUB_SP500:
                return csv
            raise ConnectionError("wikipedia down")
        u = USIndexUniverse(self._cfg(), http_get=get).get_universe()
        self.assertFalse(u.attrs["complete"])
        self.assertEqual(len(u), 3)


class TestUniverseManager(unittest.TestCase):
    def _cfg(self, path):
        cfg = make_cfg()
        cfg["universe"]["file"] = str(path)
        cfg["universe"]["sources"] = ["csv_file"]
        # production-like: CSV provider ada tetapi BUKAN provider utama
        cfg["market_data"]["providers"] = [
            {"name": "yahoo", "type": "yahoo", "enabled": True, "priority": 1},
            {**cfg["market_data"]["providers"][0], "priority": 99}]
        return cfg

    def test_file_used_and_missing_stock_deactivated(self):
        d = tmpdir()
        pd.DataFrame(SP_ROWS).to_csv(d / "u.csv", index=False)
        cfg = self._cfg(d / "u.csv")
        ctx = make_ctx(cfg)
        out = UniverseManager(cfg, ctx.repo, ctx.chain).update()
        self.assertEqual(out["source"], "file:u.csv")
        st = ctx.repo.stocks().set_index("ticker")
        self.assertEqual(sorted(st.index), ["AAPL", "BRK-B", "JPM"])
        self.assertEqual(st.loc["JPM", "sector"], "Financials")
        # saham keluar dari daftar berikutnya → nonaktif, histori tidak dihapus (rasio 2/3 < 0,8 → set ambang 0,5)
        cfg["universe"]["min_ratio_to_deactivate"] = 0.5
        pd.DataFrame(SP_ROWS[:2]).to_csv(d / "u.csv", index=False)
        out = UniverseManager(cfg, ctx.repo, ctx.chain).update()
        st = ctx.repo.stocks().set_index("ticker")
        self.assertEqual(out["deactivated"], ["JPM"])
        self.assertFalse(bool(st.loc["JPM", "is_active"]))

    def test_fallback_list_never_deactivates(self):
        """Wikipedia gagal → snapshot S&P 500 dipakai, tetapi saham mid-cap yang tidak tercantum TETAP aktif."""
        d = tmpdir()
        pd.DataFrame(SP_ROWS).to_csv(d / "u.csv", index=False)
        cfg = self._cfg(d / "u.csv")
        ctx = make_ctx(cfg)
        UniverseManager(cfg, ctx.repo, ctx.chain).update()
        pd.DataFrame(SP_ROWS[:1]).to_csv(d / "u.csv", index=False)
        cfg["universe"]["sources"] = ["us_index", "csv_file"]
        import app.data.us_universe as uu
        orig = uu._http_get
        uu.USIndexUniverse.__init__.__defaults__ = (lambda url: (_ for _ in ()).throw(ConnectionError("offline")),)
        try:
            out = UniverseManager(cfg, ctx.repo, ctx.chain).update()
        finally:
            uu.USIndexUniverse.__init__.__defaults__ = (orig,)
        self.assertEqual(out["deactivated"], [])
        self.assertTrue(ctx.repo.stocks().set_index("ticker")["is_active"].all())


class TestMigration(unittest.TestCase):
    def test_new_column_added_to_old_database(self):
        from app.database.db import Database
        from app.database.schema import existing_columns, migrate
        db = Database.from_url("sqlite:///:memory:")
        db.executescript(["CREATE TABLE stocks (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, "
                          "is_active INTEGER NOT NULL, UNIQUE (ticker))",
                          "INSERT INTO stocks (ticker, is_active) VALUES ('AAPL', 1)"])
        migrate(db)
        self.assertIn("listed_shares", existing_columns(db, "stocks"))
        self.assertEqual(db.scalar("SELECT ticker FROM stocks"), "AAPL")  # data lama utuh


if __name__ == "__main__":
    unittest.main()


class TestUniverseResolution(unittest.TestCase):
    """Regresi: daftar contoh 40 saham tidak boleh diam-diam dipakai sebagai 'seluruh universe'."""

    def _cfg(self, root):
        cfg = make_cfg()
        cfg["_root"] = str(root)
        (root / "config").mkdir(parents=True, exist_ok=True)
        cfg["universe"]["file"] = "config/universe.csv"
        return cfg

    def _write_example(self, root):
        from tests.helpers import ROOT
        (root / "config" / "universe.example.csv").write_text((ROOT / "config" / "universe.example.csv").read_text())

    def test_example_used_and_flagged_when_no_user_file(self):
        from app.data.universe_file import resolve_universe_file
        root = tmpdir()
        cfg = self._cfg(root)
        self._write_example(root)
        p, is_example = resolve_universe_file(cfg)
        self.assertEqual((p.name, is_example), ("universe.example.csv", True))

    def test_copy_of_example_under_user_name_is_still_flagged(self):
        from app.data.universe_file import resolve_universe_file
        root = tmpdir()
        cfg = self._cfg(root)
        self._write_example(root)
        (root / "config" / "universe.csv").write_text((root / "config" / "universe.example.csv").read_text())
        p, is_example = resolve_universe_file(cfg)
        self.assertEqual((p.name, is_example), ("universe.csv", True))

    def test_user_file_preferred(self):
        from app.data.universe_file import resolve_universe_file
        root = tmpdir()
        cfg = self._cfg(root)
        self._write_example(root)
        pd.DataFrame(SP_ROWS).to_csv(root / "config" / "universe.csv", index=False)
        p, is_example = resolve_universe_file(cfg)
        self.assertEqual((p.name, is_example), ("universe.csv", False))

    def test_health_warns_on_example_universe(self):
        from app.pipeline.health import health_check
        root = tmpdir()
        cfg = self._cfg(root)
        self._write_example(root)
        cfg["database"]["url"] = f"sqlite:///{root / 'h.db'}"
        cfg["market_data"]["providers"] = [{"name": "yahoo", "type": "yahoo", "enabled": True, "priority": 1},
                                           {**make_cfg()["market_data"]["providers"][0], "priority": 99}]
        ctx = make_ctx(cfg)
        UniverseManager(cfg, ctx.repo, ctx.chain).update()
        h = health_check(ctx, check_provider=False)
        self.assertEqual(h["checks"]["UNIVERSE"]["status"], "WARN")
        self.assertIn("DAFTAR CONTOH", h["checks"]["UNIVERSE"]["detail"])
        pd.DataFrame(SP_ROWS).to_csv(root / "config" / "universe.csv", index=False)
        UniverseManager(cfg, ctx.repo, ctx.chain).update()
        self.assertEqual(health_check(ctx, check_provider=False)["checks"]["UNIVERSE"]["status"], "OK")
