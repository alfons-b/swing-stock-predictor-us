import unittest

import pandas as pd

from app.database.db import Database
from app.database.repository import Repository
from app.database.schema import TABLE_NAMES, migrate
from tests.helpers import db_urls


class TestUpsert(unittest.TestCase):
    def test_idempotent_insert_update(self):
        for url in db_urls():
            with self.subTest(url=url.split("@")[-1]):
                db = Database.from_url(url)
                if db.dialect == "postgres":
                    db.executescript([f"DROP TABLE IF EXISTS {t} CASCADE" for t in reversed(TABLE_NAMES)])
                migrate(db)
                r = Repository(db)
                r.upsert_stocks(pd.DataFrame({"ticker": ["AAAA", "BBBB"], "is_active": [True, True]}))
                sid = r.stock_ids()["AAAA"]
                p = pd.DataFrame({"stock_id": sid, "date": pd.bdate_range("2024-01-01", periods=5), "open": 1.0, "high": 2.0,
                                  "low": 0.5, "close": 1.5, "volume": 10.0, "value": 15.0, "is_adjusted": False, "source": "t"})
                self.assertEqual(r.upsert_prices(p), (5, 0))
                self.assertEqual(r.upsert_prices(p), (0, 5))          # dijalankan ulang → tidak ada duplikat
                p2 = pd.concat([p.tail(2).assign(close=1.9), p.head(1).assign(date=pd.Timestamp("2024-01-08"))])
                self.assertEqual(r.upsert_prices(p2), (1, 2))
                c = r.price_counts()
                self.assertEqual((c["rows"], c["duplicates"]), (6, 0))
                self.assertAlmostEqual(float(db.scalar("SELECT close FROM price_history WHERE date = ?", ("2024-01-05",))), 1.9)
                db.close()

    def test_duplicate_keys_in_batch_collapsed(self):
        db = Database.from_url("sqlite:///:memory:")
        migrate(db)
        r = Repository(db)
        r.upsert_stocks(pd.DataFrame({"ticker": ["AAAA", "AAAA"], "is_active": [True, False]}))
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM stocks"), 1)
        self.assertFalse(bool(db.scalar("SELECT is_active FROM stocks")))


if __name__ == "__main__":
    unittest.main()
