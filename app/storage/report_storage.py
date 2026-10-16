"""ReportStorage (§43). Runner GitHub Actions ephemeral → laporan production disimpan ke cloud.

- LocalReportStorage    : reports/ + data/output/ (development; juga di-upload sebagai artifact Actions)
- DatabaseReportStorage : tabel `reports` di Supabase (default production) — dibaca dashboard
- SupabaseReportStorage : Supabase Storage bucket
Semua backend juga menulis salinan lokal agar workflow bisa meng-upload artifact.
"""
from __future__ import annotations

import os
from pathlib import Path

from app.config import get, resolve_path
from app.database.repository import Repository, now_utc

CONTENT_TYPES = {".html": "text/html", ".csv": "text/csv", ".md": "text/markdown", ".json": "application/json"}


class ReportStorage:
    backend = "base"

    def __init__(self, cfg: dict, repo: Repository):
        self.cfg, self.repo = cfg, repo

    def local_path(self, kind: str, filename: str) -> Path:
        key = "storage.local_output_dir" if kind == "predictions_csv" else "storage.local_report_dir"
        d = resolve_path(self.cfg, get(self.cfg, key, "reports"))
        d.mkdir(parents=True, exist_ok=True)
        return d / filename

    def save(self, report_date: str, kind: str, filename: str, content: bytes, run_id: str | None = None) -> str:
        path = self.local_path(kind, filename)
        path.write_bytes(content)
        uri, inline = self._put(filename, content, path)
        self.repo.db.upsert("reports", [{"report_date": report_date, "kind": kind, "filename": filename, "content": inline,
                                         "content_type": CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
                                         "storage_backend": self.backend, "storage_uri": uri, "run_id": run_id,
                                         "created_at": now_utc()}], keys=["report_date", "kind", "filename"], count=False)
        return uri

    def _put(self, filename, content, path):
        return str(path), None

    def latest(self, kind: str) -> tuple[str, bytes] | None:
        rows = self.repo.db.query("SELECT filename, content, storage_backend, storage_uri FROM reports WHERE kind = ? "
                                  "ORDER BY report_date DESC, created_at DESC LIMIT 1", (kind,))
        if not rows:
            return None
        fn, content, backend, uri = rows[0]
        if content is not None:
            return fn, bytes(content)
        if backend == "supabase":
            return fn, SupabaseReportStorage(self.cfg, self.repo).fetch(uri)
        p = Path(uri)
        return (fn, p.read_bytes()) if p.exists() else None


class LocalReportStorage(ReportStorage):
    backend = "local"


class DatabaseReportStorage(ReportStorage):
    backend = "database"

    def _put(self, filename, content, path):
        return f"db://reports/{filename}", content


class SupabaseReportStorage(ReportStorage):
    backend = "supabase"

    def __init__(self, cfg, repo):
        super().__init__(cfg, repo)
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.key = os.environ.get("SUPABASE_KEY", "")
        self.bucket = get(cfg, "storage.supabase_bucket", "swing-artifacts")
        if not (self.url and self.key):
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY belum diset")

    def _put(self, filename, content, path):
        import requests
        obj = f"reports/{filename}"
        r = requests.post(f"{self.url}/storage/v1/object/{self.bucket}/{obj}", data=content, timeout=120,
                          headers={"Authorization": f"Bearer {self.key}", "apikey": self.key, "x-upsert": "true",
                                   "Content-Type": CONTENT_TYPES.get(path.suffix, "application/octet-stream")})
        if r.status_code >= 300:
            raise RuntimeError(f"Upload report gagal: HTTP {r.status_code}")
        return f"supabase://{self.bucket}/{obj}", None

    def fetch(self, uri: str) -> bytes:
        import requests
        obj = uri.split(f"{self.bucket}/", 1)[-1]
        r = requests.get(f"{self.url}/storage/v1/object/{self.bucket}/{obj}", timeout=60,
                         headers={"Authorization": f"Bearer {self.key}", "apikey": self.key})
        r.raise_for_status()
        return r.content


def make_report_storage(cfg: dict, repo: Repository) -> ReportStorage:
    b = str(get(cfg, "storage.report_backend", "auto")).lower()
    if b == "auto":
        b = "database" if repo.db.dialect == "postgres" else "local"
    return {"local": LocalReportStorage, "database": DatabaseReportStorage, "supabase": SupabaseReportStorage}[b](cfg, repo)
