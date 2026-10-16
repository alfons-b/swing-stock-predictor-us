"""AppContext (config + DB + provider + kalender + storage) dan PipelineRun (§21, §54).

PipelineRun:
- menulis baris pipeline_runs (RUNNING) di awal, lalu SUCCESS / PARTIAL_SUCCESS / FAILED di akhir;
- menyalin log WARNING+ dan log bertanda `extra={"persist": True}` ke system_logs;
- exception tetap dilempar ulang → proses exit code != 0 → GitHub Actions berstatus FAILED.
"""
from __future__ import annotations

import logging
import os
import traceback
import uuid
from dataclasses import dataclass, field

from app.config import get
from app.database.db import Database
from app.database.repository import Repository, now_utc
from app.database.schema import migrate
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


# Provider yang dipakai proyek LAIN milik pemilik repo (versi BEI memakai Neon, aplikasi lain memakai Supabase).
# Produksi versi US tidak boleh menumpang di sana. Override sadar per provider lewat env.
RESERVED_PROVIDERS = {"supabase": "ALLOW_SUPABASE_DATABASE", "neon": "ALLOW_NEON_DATABASE"}


def _allowed(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes")


def guard_production_database(cfg: dict, db: Database) -> None:
    """Aturan proyek: database produksi TIDAK boleh Supabase atau Neon (dipakai proyek lain milik pemilik repo).

    Dicek SEBELUM migrasi, agar secret yang salah tempel gagal jelas (exit 2) alih-alih membuat atau mengubah
    tabel di database proyek lain. Override sadar: ALLOW_SUPABASE_DATABASE=true / ALLOW_NEON_DATABASE=true.
    Lihat docs/DATABASE_DECISION.md."""
    from app.config import ConfigError, is_production
    provider = getattr(db, "provider", None)
    if provider not in RESERVED_PROVIDERS or not is_production(cfg):
        return
    env = RESERVED_PROVIDERS[provider]
    if _allowed(env):
        log.warning("DATABASE_URL menunjuk ke %s dan %s=true — melanjutkan atas izin eksplisit", provider, env)
        return
    raise ConfigError(f"DATABASE_URL menunjuk ke {provider.capitalize()}, yang dipakai proyek lain Anda. Versi US harus "
                      "memakai database terpisah (Aiven for PostgreSQL Free). Ganti secret DATABASE_URL dengan "
                      "service URI Aiven lalu jalankan workflow Initial Setup (docs/SETUP_GUIDE.md).")


@dataclass
class AppContext:
    cfg: dict
    db: Database
    repo: Repository
    _chain: object = None
    _calendar: object = None
    _models: object = None
    _reports: object = None

    @classmethod
    def create(cls, cfg: dict, migrate_db: bool = True) -> "AppContext":
        db = Database.from_url(get(cfg, "database.url"), root=cfg.get("_root"))
        guard_production_database(cfg, db)
        if migrate_db:
            migrate(db)
        return cls(cfg, db, Repository(db))

    @property
    def chain(self):
        if self._chain is None:
            from app.data.providers.chain import ProviderChain
            self._chain = ProviderChain(self.cfg)
        return self._chain

    @property
    def calendar(self):
        if self._calendar is None:
            from app.market.calendar import TradingCalendar
            self._calendar = TradingCalendar(self.cfg, self.repo.index_dates(get(self.cfg, "data.index_id", "COMPOSITE")))
        return self._calendar

    @property
    def models(self):
        if self._models is None:
            from app.storage.model_storage import make_model_storage
            self._models = make_model_storage(self.cfg, self.repo)
        return self._models

    @property
    def reports(self):
        if self._reports is None:
            from app.storage.report_storage import make_report_storage
            self._reports = make_report_storage(self.cfg, self.repo)
        return self._reports

    def is_synthetic(self) -> bool:
        try:
            return self.chain.is_synthetic
        except Exception:
            return False

    def close(self):
        self.db.close()


class _DBLogHandler(logging.Handler):
    def __init__(self, run_id: str):
        super().__init__(level=logging.INFO)
        self.run_id = run_id
        self.rows: list[dict] = []

    def emit(self, record):
        if record.levelno >= logging.WARNING or getattr(record, "persist", False):
            self.rows.append({"run_id": self.run_id, "ts": now_utc(), "level": record.levelname,
                              "logger": record.name, "message": record.getMessage()})


@dataclass
class PipelineRun:
    ctx: AppContext
    run_type: str
    run_id: str = ""
    stats: dict = field(default_factory=dict)
    status: str = "RUNNING"

    def __post_init__(self):
        gh = os.environ.get("GITHUB_RUN_ID")
        self.run_id = self.run_id or f"{self.run_type}-{gh or uuid.uuid4().hex[:10]}-{now_utc()[:19].replace(':', '')}"
        self.trigger = os.environ.get("GITHUB_EVENT_NAME", "local")

    def __enter__(self):
        self.ctx.repo.start_run(self.run_id, self.run_type, self.trigger)
        self.handler = _DBLogHandler(self.run_id)
        logging.getLogger().addHandler(self.handler)
        app_logger = logging.getLogger("app")
        self._prev_level = app_logger.level
        if app_logger.getEffectiveLevel() > logging.INFO:  # event penting harus tercatat walau logging belum dikonfigurasi
            app_logger.setLevel(logging.INFO)
        log.info("Pipeline %s dimulai (%s, trigger=%s, db=%s)", self.run_type, self.run_id, self.trigger,
                 self.ctx.db.describe(), extra={"persist": True})
        return self

    def __exit__(self, exc_type, exc, tb):
        logging.getLogger().removeHandler(self.handler)
        logging.getLogger("app").setLevel(self._prev_level)
        fields = {k: self.stats.get(k) for k in ("stocks_processed", "stocks_failed", "rows_inserted", "rows_updated",
                                                 "model_version", "data_status", "market_date") if k in self.stats}
        if exc is not None:
            self.status = "FAILED"
            fields["error_message"] = f"{exc_type.__name__}: {exc}"[:2000]
            self.handler.rows.append({"run_id": self.run_id, "ts": now_utc(), "level": "ERROR", "logger": "pipeline",
                                      "message": "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]})
        elif self.status == "RUNNING":
            self.status = "SUCCESS"
        try:
            self.ctx.repo.finish_run(self.run_id, status=self.status, summary=self.stats.get("summary"), **fields)
            self.ctx.repo.write_logs(self.handler.rows)
        except Exception as e:  # jangan menutupi error asli
            log.error("Gagal menulis status pipeline: %s", e)
        log.info("Pipeline %s selesai: %s", self.run_type, self.status)
        return False  # exception diteruskan → exit code != 0
