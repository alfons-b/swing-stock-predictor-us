"""Profil provider database (Neon/Aiven/Supabase) — deteksi, SSL, batas ukuran, pemulihan sebelum penulisan diblokir."""
import unittest

from app.database.db import Database
from app.database.maintenance import size_status
from app.database.providers import (detect_provider, is_pooled_neon, normalize_url, profile_for,
                                    resolve_size_limit_mb)
from tests.helpers import make_cfg

NEON = "postgresql://app:secret@ep-cool-cloud-a1b2c3d4.ap-southeast-1.aws.neon.tech/swing"
NEON_POOLED = "postgresql://app:secret@ep-cool-cloud-a1b2c3d4-pooler.ap-southeast-1.aws.neon.tech/swing?sslmode=require"
AIVEN = "postgres://avnadmin:secret@pg-swing-alfons.f.aivencloud.com:12345/defaultdb?sslmode=require"
SUPA = "postgresql://postgres.abc:secret@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"


class TestDetectProvider(unittest.TestCase):
    def test_detects_known_hosts(self):
        self.assertEqual(detect_provider(NEON), "neon")
        self.assertEqual(detect_provider(NEON_POOLED), "neon")
        self.assertEqual(detect_provider(AIVEN), "aiven")
        self.assertEqual(detect_provider(SUPA), "supabase")
        self.assertEqual(detect_provider("postgresql://ci:ci@localhost:5432/t"), "local_postgres")
        self.assertEqual(detect_provider("postgresql://u:p@db.example.org/x"), "postgres")
        self.assertEqual(detect_provider("sqlite:///:memory:"), "sqlite")
        self.assertEqual(detect_provider(None), "sqlite")

    def test_pooler_detection(self):
        self.assertTrue(is_pooled_neon(NEON_POOLED))
        self.assertFalse(is_pooled_neon(NEON))


class TestNormalizeUrl(unittest.TestCase):
    def test_adds_sslmode_for_remote_hosts(self):
        self.assertIn("sslmode=require", normalize_url(NEON))
        self.assertIn("sslmode=require", normalize_url("postgresql://u:p@db.example.org/x"))

    def test_never_overrides_user_sslmode(self):
        u = NEON + "?sslmode=verify-full&channel_binding=require"
        out = normalize_url(u)
        self.assertIn("sslmode=verify-full", out)
        self.assertNotIn("sslmode=require", out)
        self.assertIn("channel_binding=require", out)

    def test_local_postgres_and_sqlite_untouched(self):
        self.assertEqual(normalize_url("postgresql://ci:ci@localhost:5432/t"), "postgresql://ci:ci@localhost:5432/t")
        self.assertEqual(normalize_url("sqlite:///x.db"), "sqlite:///x.db")

    def test_password_preserved_and_not_in_describe(self):
        db = Database.from_url(NEON)                       # tidak konek (lazy)
        self.assertIn("secret", db.url)                    # password tetap dipakai untuk konek
        self.assertNotIn("secret", db.describe())
        self.assertIn("neon", db.describe())
        self.assertEqual(db.provider, "neon")


class TestSizeLimit(unittest.TestCase):
    def test_auto_uses_provider_profile(self):
        self.assertEqual(resolve_size_limit_mb("auto", NEON), profile_for(NEON).free_size_limit_mb)
        self.assertEqual(resolve_size_limit_mb(None, SUPA), 500)
        self.assertLess(resolve_size_limit_mb("auto", NEON), 1024)        # di bawah batas resmi 1 GB
        self.assertLess(resolve_size_limit_mb("auto", AIVEN), 1024)

    def test_explicit_value_wins(self):
        self.assertEqual(resolve_size_limit_mb("8000", NEON), 8000.0)
        self.assertEqual(resolve_size_limit_mb(250, NEON), 250.0)

    def test_size_status_reports_provider(self):
        db = Database.from_url("sqlite:///:memory:")
        st = size_status(make_cfg(), db)
        self.assertEqual(st["provider"], "sqlite")


class TestEmergencyMaintenance(unittest.TestCase):
    """Neon tidak menandai read-only; penulisan langsung gagal saat kuota habis → pulihkan pada rasio ≥ 95%."""

    def _ctx(self):
        from app.pipeline.context import AppContext
        cfg = make_cfg(as_of="2026-10-09")
        return AppContext.create(cfg)

    def test_triggers_on_ratio_without_read_only_flag(self):
        from app.pipeline import jobs
        ctx = self._ctx()
        called = {}

        def fake_maint(cfg, db, today, dry_run=False, full=False):
            called["full"] = full
            return {"after_mb": 0.0}

        size_mb = size_status(ctx.cfg, ctx.db)["bytes"] / 1e6
        ctx.cfg["database"]["size_limit_mb"] = size_mb / 0.97              # 97% dari batas, read_only False
        import app.database.maintenance as m
        real = m.run_maintenance                                           # ensure_writable mengimpor saat dipanggil
        m.run_maintenance = fake_maint
        try:
            jobs.ensure_writable(ctx)
        finally:
            m.run_maintenance = real
        self.assertEqual(called.get("full"), True)
        ctx.close()

    def test_no_action_below_threshold(self):
        from app.pipeline import jobs
        import app.database.maintenance as m
        ctx = self._ctx()
        ctx.cfg["database"]["size_limit_mb"] = 1e6
        real = m.run_maintenance
        m.run_maintenance = lambda *a, **k: self.fail("maintenance tidak boleh jalan")
        try:
            jobs.ensure_writable(ctx)
        finally:
            m.run_maintenance = real
        ctx.close()

    def test_aborts_clearly_when_still_full(self):
        from app.pipeline.jobs import PipelineAbort, ensure_writable
        ctx = self._ctx()
        ctx.cfg["database"]["size_limit_mb"] = 0.0001                      # tidak mungkin turun di bawah batas
        with self.assertRaises(PipelineAbort) as e:
            ensure_writable(ctx)
        self.assertIn("DB_SIZE_LIMIT_MB", str(e.exception))
        ctx.close()


if __name__ == "__main__":
    unittest.main()


class TestStrictHealth(unittest.TestCase):
    def test_write_probe_ok_and_cleans_up(self):
        from app.pipeline.health import write_probe
        from app.pipeline.context import AppContext
        ctx = AppContext.create(make_cfg())
        self.assertEqual(write_probe(ctx.db)[0], "OK")
        self.assertEqual(ctx.db.scalar("SELECT COUNT(*) FROM system_logs WHERE run_id = 'health-write-probe'"), 0)
        ctx.close()

    def test_write_probe_detects_blocked_storage(self):
        from app.pipeline.health import write_probe
        from app.pipeline.context import AppContext
        ctx = AppContext.create(make_cfg())
        ctx.db.conn.execute("PRAGMA query_only = ON")                   # meniru kuota habis / read-only
        st, detail = write_probe(ctx.db)
        self.assertEqual(st, "FAIL")
        self.assertIn("penulisan gagal", detail)
        ctx.close()

    def test_strict_failures(self):
        from app.pipeline.health import strict_failures
        h = {"checks": {"DATABASE": {"status": "OK"}, "DATABASE SIZE": {"status": "FAIL"},
                        "DATA FRESHNESS": {"status": "FAIL"}, "DATABASE WRITE": {"status": "FAIL"}}}
        self.assertEqual(strict_failures(h), ["DATABASE SIZE", "DATABASE WRITE"])   # freshness bukan urusan DB check

    def test_cli_strict_exit_code(self):
        import main
        from tests.helpers import make_cfg as mk
        cfg = mk()
        url = cfg["database"]["url"]
        import os
        old = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = url
        try:
            self.assertEqual(main.main(["health", "--no-provider", "--strict"]), 0)
        finally:
            if old is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = old


class TestReservedDatabaseGuard(unittest.TestCase):
    """Produksi versi US tidak boleh Supabase ATAU Neon (dipakai proyek lain); guard berjalan SEBELUM migrasi."""

    def _cfg(self, env, url=SUPA):
        cfg = make_cfg(db_url=url)
        cfg["environment"] = env
        return cfg

    def test_production_supabase_and_neon_refused_before_connecting(self):
        import os
        from app.config import ConfigError
        from app.pipeline.context import AppContext
        for url, env in ((SUPA, "ALLOW_SUPABASE_DATABASE"), (NEON, "ALLOW_NEON_DATABASE")):
            os.environ.pop(env, None)
            with self.assertRaises(ConfigError) as e:
                AppContext.create(self._cfg("production", url))   # akan gagal konek bila guard tidak berjalan dulu
            self.assertIn("Aiven", str(e.exception))

    def test_explicit_override_and_non_production(self):
        import os
        from app.pipeline.context import guard_production_database
        for url, env in ((SUPA, "ALLOW_SUPABASE_DATABASE"), (NEON, "ALLOW_NEON_DATABASE")):
            db = Database.from_url(url)
            guard_production_database(self._cfg("development", url), db)   # dev: tidak dicegah
            os.environ[env] = "true"
            try:
                guard_production_database(self._cfg("production", url), db)
            finally:
                os.environ.pop(env, None)

    def test_aiven_production_allowed(self):
        from app.pipeline.context import guard_production_database
        guard_production_database(self._cfg("production", AIVEN), Database.from_url(AIVEN))