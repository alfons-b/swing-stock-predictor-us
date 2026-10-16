"""Ketahanan koneksi database untuk job panjang di Neon.

Insiden 12 Okt 2026 (Initial Setup di Neon): query baca membuka transaksi implisit (psycopg, autocommit off) yang
tidak pernah ditutup; selama ±6 menit fitur dihitung di CPU, sesi berstatus 'idle in transaction' → Neon memutus
koneksi (IdleInTransactionSessionTimeout) → retrain gagal. Test ini menjaga:
  1. setelah query baca, sesi TIDAK lagi berada di dalam transaksi (kecuali di dalam db.transaction());
  2. koneksi yang diputus server dipulihkan otomatis SEKALI untuk operasi di luar transaksi eksplisit;
  3. di dalam transaksi eksplisit, koneksi putus TIDAK disembunyikan (atomisitas lebih penting).
"""
import os
import time
import unittest

from app.database.db import Database


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = [("x",)]
        self.rowcount = 1

    def execute(self, sql, params=()):
        self.conn.calls.append(sql)
        if self.conn.fail_next:
            self.conn.fail_next = False
            self.conn.closed = True
            raise ConnectionError("server closed the connection unexpectedly")
        self.conn.in_tx = True                  # perilaku psycopg: statement pertama membuka transaksi

    def executemany(self, sql, rows):
        self.execute(sql)

    def fetchall(self):
        return [(1,)]


class _FakeConn:
    def __init__(self, owner):
        self.owner = owner
        self.in_tx = False
        self.closed = False
        self.fail_next = False
        self.calls = []

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        if self.closed:
            raise ConnectionError("closed")
        self.in_tx = False

    def rollback(self):
        if self.closed:
            raise ConnectionError("closed")
        self.in_tx = False

    def close(self):
        self.closed = True


class _FakeDB(Database):
    """Database dengan koneksi tiruan bersemantik psycopg (transaksi implisit, koneksi bisa diputus)."""
    dialect = "fake"
    placeholder = "?"

    def __init__(self):
        super().__init__("fake://")
        self.connects = 0
        self.fail_on_connect = []

    def connect(self):
        self.connects += 1
        c = _FakeConn(self)
        if self.fail_on_connect:
            c.fail_next = self.fail_on_connect.pop(0)
        return c


class TestReadsCloseTransaction(unittest.TestCase):
    def test_query_methods_leave_no_open_transaction(self):
        db = _FakeDB()
        db.query("SELECT 1")
        self.assertFalse(db.conn.in_tx)
        db.query_df("SELECT 1")
        self.assertFalse(db.conn.in_tx)
        db.scalar("SELECT 1")
        self.assertFalse(db.conn.in_tx)

    def test_reads_inside_explicit_transaction_do_not_commit(self):
        db = _FakeDB()
        with db.transaction():
            db.conn.cursor().execute("UPDATE t SET a = 1")
            db.query("SELECT 1")
            self.assertTrue(db.conn.in_tx)             # atomisitas tetap: belum di-commit
        self.assertFalse(db.conn.in_tx)


class TestSqlTranslation(unittest.TestCase):
    def test_postgres_escapes_literal_percent(self):
        db = Database.from_url("postgresql://u:p@localhost:5432/x")      # tidak konek (lazy)
        self.assertEqual(db._sql("SELECT * FROM t WHERE a LIKE 'x-%' AND b = ?"),
                         "SELECT * FROM t WHERE a LIKE 'x-%%' AND b = %s")
        self.assertEqual(Database.from_url("sqlite:///:memory:")._sql("a LIKE 'x-%' AND b = ?"), "a LIKE 'x-%' AND b = ?")


class TestReconnect(unittest.TestCase):
    def test_dropped_connection_is_reopened_once(self):
        db = _FakeDB()
        db.query("SELECT 1")
        db.conn.fail_next = True                       # server memutus koneksi saat query berikutnya
        self.assertEqual(db.query("SELECT 2"), [(1,)])
        self.assertEqual(db.connects, 2)

    def test_reconnect_also_for_writes_outside_transaction(self):
        db = _FakeDB()
        db.query("SELECT 1")
        db.conn.fail_next = True
        db.execute("INSERT INTO t VALUES (1) ON CONFLICT DO NOTHING")
        self.assertEqual(db.connects, 2)

    def test_gives_up_after_one_retry(self):
        db = _FakeDB()
        db.fail_on_connect = [True, True]              # koneksi baru juga langsung putus
        with self.assertRaises(ConnectionError):
            db.query("SELECT 1")
        self.assertEqual(db.connects, 2)

    def test_no_silent_retry_inside_explicit_transaction(self):
        db = _FakeDB()
        db.query("SELECT 1")
        with self.assertRaises(ConnectionError):
            with db.transaction():
                db.conn.cursor().execute("UPDATE t SET a = 1")
                db.conn.fail_next = True
                db.query("SELECT 1")
        self.assertEqual(db.connects, 1)


@unittest.skipUnless(os.environ.get("TEST_POSTGRES_URL"), "butuh TEST_POSTGRES_URL (CI: service PostgreSQL 16)")
class TestPostgresLongJob(unittest.TestCase):
    """Reproduksi insiden Neon di PostgreSQL sungguhan."""

    def setUp(self):
        self.db = Database.from_url(os.environ["TEST_POSTGRES_URL"])

    def tearDown(self):
        self.db.close()

    def test_survives_idle_in_transaction_timeout(self):
        db = self.db
        db.execute("SET idle_in_transaction_session_timeout = '1s'")
        db.query_df("SELECT 1 AS x")                  # baca → sebelum perbaikan: sesi tertinggal 'idle in transaction'
        import psycopg
        self.assertEqual(db.conn.info.transaction_status, psycopg.pq.TransactionStatus.IDLE)
        time.sleep(2.0)                                # meniru hitung fitur ±6 menit di CPU
        self.assertEqual(db.scalar("SELECT 2"), 2)

    def test_literal_percent_in_sql(self):
        db = self.db
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM (SELECT 'wf-test-1' AS r) t WHERE r LIKE 'wf-test-%'"), 1)
        self.assertEqual(db.scalar("SELECT COUNT(*) FROM (SELECT 'wf-test-1' AS r) t WHERE r LIKE ?", ("wf-%",)), 1)
        self.assertEqual(db.scalar("SELECT '100%'"), "100%")

    def test_recovers_when_server_terminates_session(self):
        db = self.db
        pid = db.scalar("SELECT pg_backend_pid()")
        admin = Database.from_url(os.environ["TEST_POSTGRES_URL"])
        try:
            admin.scalar("SELECT pg_terminate_backend(?)", (pid,))
        finally:
            admin.close()
        time.sleep(0.5)
        self.assertEqual(db.scalar("SELECT 3"), 3)    # koneksi baru dibuka otomatis
        self.assertNotEqual(db.scalar("SELECT pg_backend_pid()"), pid)


if __name__ == "__main__":
    unittest.main()
