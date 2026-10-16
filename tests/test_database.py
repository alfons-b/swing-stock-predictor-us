import unittest
import uuid

import pandas as pd

from app.database.db import Database
from app.database.repository import Repository
from app.database.schema import TABLE_NAMES, migrate, render_ddl
from tests.helpers import db_urls

REQUIRED = ["stocks", "price_history", "corporate_actions", "market_index", "sector_data", "features", "predictions",
            "prediction_evaluations", "trading_signals", "backtest_runs", "backtest_trades", "model_versions",
            "pipeline_runs", "data_sources", "system_logs"]


def fresh(url):
    db = Database.from_url(url)
    if db.dialect == "postgres":  # bersihkan skema test
        db.executescript([f"DROP TABLE IF EXISTS {t} CASCADE" for t in reversed(TABLE_NAMES)])
    migrate(db)
    return db


class TestDatabase(unittest.TestCase):
    def test_required_tables_exist_and_migration_idempotent(self):
        self.assertTrue(set(REQUIRED) <= set(TABLE_NAMES))
        for url in db_urls():
            with self.subTest(url=url.split("@")[-1]):
                db = fresh(url)
                migrate(db)
                migrate(db)
                self.assertEqual(db.scalar("SELECT COUNT(*) FROM schema_migrations"), 1)
                for t in REQUIRED:
                    db.scalar(f"SELECT COUNT(*) FROM {t}")
                db.close()

    def test_unique_constraint_stock_date(self):
        for url in db_urls():
            with self.subTest(url=url.split("@")[-1]):
                db = fresh(url)
                r = Repository(db)
                r.upsert_stocks(pd.DataFrame({"ticker": ["AAAA"], "is_active": [True]}))
                sid = r.stock_ids()["AAAA"]
                db.execute("INSERT INTO price_history (stock_id, date, close) VALUES (?, ?, ?)", (sid, "2024-01-02", 100.0))
                with self.assertRaises(Exception):
                    db.execute("INSERT INTO price_history (stock_id, date, close) VALUES (?, ?, ?)", (sid, "2024-01-02", 101.0))
                db.conn.rollback()
                db.close()

    def test_blob_roundtrip(self):
        for url in db_urls():
            with self.subTest(url=url.split("@")[-1]):
                db = fresh(url)
                blob = uuid.uuid4().bytes * 1000
                db.upsert("model_versions", [{"version": "model_v001", "status": "CANDIDATE", "artifact": blob}], keys=["version"])
                self.assertEqual(bytes(db.scalar("SELECT artifact FROM model_versions WHERE version = ?", ("model_v001",))), blob)
                db.close()

    def test_postgres_ddl_renders(self):
        ddl = "\n".join(render_ddl("postgres"))
        self.assertIn("BIGSERIAL PRIMARY KEY", ddl)
        self.assertIn("UNIQUE (stock_id, date)", ddl)
        self.assertIn("BYTEA", ddl)
        self.assertNotIn("AUTOINCREMENT", ddl)

    def test_describe_hides_password(self):
        from app.database.db import PostgresDatabase
        d = PostgresDatabase("postgresql://user:SECRET@host:5432/db").describe()
        self.assertNotIn("SECRET", d)
        self.assertNotIn("user", d)


if __name__ == "__main__":
    unittest.main()


class TestSchemaConflict(unittest.TestCase):
    def test_foreign_table_detected_with_clear_message(self):
        """Regresi: data/local.db yang dibuat program lain → pesan jelas, bukan 'NOT NULL constraint failed'."""
        from app.database.schema import SchemaConflictError
        for url in db_urls():
            with self.subTest(url=url.split("@")[-1]):
                db = Database.from_url(url)
                db.executescript([f"DROP TABLE IF EXISTS {t} CASCADE" if db.dialect == "postgres" else f"DROP TABLE IF EXISTS {t}"
                                  for t in reversed(TABLE_NAMES)])
                db.executescript(["CREATE TABLE pipeline_runs (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, "
                                  "stocks_processed INTEGER NOT NULL, foreign_col TEXT NOT NULL)"])
                with self.assertRaises(SchemaConflictError) as cm:
                    migrate(db)
                msg = str(cm.exception)
                self.assertIn("pipeline_runs.stocks_processed", msg)
                self.assertIn("pipeline_runs.foreign_col", msg)
                db.executescript(["DROP TABLE pipeline_runs"])
                migrate(db)  # setelah tabel asing dihapus, migrasi normal
                db.close()


class FailedStatementRecoveryTest(unittest.TestCase):
    """Query gagal tidak boleh membuat koneksi berumur panjang (dashboard) macet di InFailedSqlTransaction."""

    def test_connection_usable_after_error(self):
        from app.database.db import Database
        for url in db_urls():
            with self.subTest(db=url.split(":")[0]):
                db = Database.from_url(url)
                with self.assertRaises(Exception):
                    db.query("SELECT * FROM tabel_yang_tidak_ada")
                self.assertEqual(db.scalar("SELECT 1"), 1)
                with self.assertRaises(Exception):
                    db.query_df("SELECT kolom_tidak_ada FROM tabel_yang_tidak_ada")
                self.assertEqual(len(db.query_df("SELECT 1 AS x")), 1)
                db.close()
