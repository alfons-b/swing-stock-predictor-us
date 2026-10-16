"""Persistent model storage (§13) + versioning (§14).

Registry (versi, status, metrik, config hash, periode training) SELALU di tabel `model_versions`
— database adalah sumber kebenaran. File model disimpan oleh backend:

- LocalModelStorage     : folder models/ (development; runner GitHub Actions ephemeral → JANGAN untuk production)
- DatabaseModelStorage  : kolom BYTEA di Supabase Postgres (default production: cukup DATABASE_URL)
- SupabaseModelStorage  : Supabase Storage bucket (CloudModelStorage; untuk model besar)

Status: CANDIDATE → ACTIVE (promote) | REJECTED ; ACTIVE lama → RETIRED. Tidak ada overwrite.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from app.config import cache_dir, get, resolve_path
from app.database.repository import Repository, now_utc
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


class ModelStorageError(RuntimeError):
    pass


def serialize(model) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.getvalue()


def deserialize(blob: bytes):
    return joblib.load(io.BytesIO(bytes(blob)))


class ModelStorage:
    backend = "base"

    def __init__(self, cfg: dict, repo: Repository):
        self.cfg, self.repo, self.db = cfg, repo, repo.db

    # ----- backend-specific
    def _put(self, version: str, blob: bytes) -> tuple[str, bytes | None]:
        raise NotImplementedError

    def _get(self, row: dict) -> bytes:
        raise NotImplementedError

    # ----- registry
    def next_version(self) -> str:
        rows = self.db.query("SELECT version FROM model_versions")
        nums = [int(v[0].split("_v")[-1]) for v in rows if "_v" in v[0] and v[0].split("_v")[-1].isdigit()]
        return f"model_v{(max(nums) + 1) if nums else 1:03d}"

    def save(self, model, metadata: dict, status: str = "CANDIDATE") -> str:
        version = self.next_version()
        model.meta["model_version"] = version
        blob = serialize(model)
        sha = hashlib.sha256(blob).hexdigest()
        uri, inline = self._put(version, blob)
        row = {"version": version, "status": status, "created_at": now_utc(),
               "train_start": metadata.get("train_start"), "train_end": metadata.get("train_end"),
               "feature_version": metadata.get("feature_version"), "config_hash": metadata.get("config_hash"),
               "classifiers": ",".join(metadata.get("classifiers", [])),
               "metrics": json.dumps(metadata.get("metrics", {}), default=str),
               "storage_backend": self.backend, "storage_uri": uri, "artifact": inline, "artifact_sha256": sha,
               "artifact_bytes": len(blob), "notes": metadata.get("notes")}
        self.db.upsert("model_versions", [row], keys=["version"], count=False)
        log.info("Model %s disimpan (%s, %.1f MB, status %s)", version, self.backend, len(blob) / 1e6, status)
        return version

    def list_versions(self) -> pd.DataFrame:
        return self.db.query_df("SELECT version, status, created_at, promoted_at, train_start, train_end, feature_version, "
                                "config_hash, classifiers, storage_backend, artifact_bytes, metrics, notes "
                                "FROM model_versions ORDER BY created_at DESC, version DESC")

    def active_version(self) -> dict | None:
        df = self.db.query_df("SELECT version, status, created_at, promoted_at, train_end, metrics, classifiers, "
                              "feature_version, config_hash FROM model_versions WHERE status = 'ACTIVE' "
                              "ORDER BY promoted_at DESC LIMIT 1")
        if not len(df):
            return None
        r = df.iloc[0].to_dict()
        r["metrics"] = json.loads(r["metrics"]) if r.get("metrics") else {}
        return r

    def load(self, version: str):
        df = self.db.query_df("SELECT * FROM model_versions WHERE version = ?", (version,))
        if not len(df):
            raise ModelStorageError(f"Model {version} tidak ada di registry")
        row = df.iloc[0].to_dict()
        cache = cache_dir(self.cfg) / "models" / f"{version}.joblib"
        if cache.exists() and hashlib.sha256(cache.read_bytes()).hexdigest() == row["artifact_sha256"]:
            blob = cache.read_bytes()
        else:
            blob = bytes(self._backend_for(row)._get(row))
            if hashlib.sha256(blob).hexdigest() != row["artifact_sha256"]:
                raise ModelStorageError(f"Checksum model {version} tidak cocok — artefak rusak")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(blob)
        model = deserialize(blob)
        model.meta["model_version"] = version
        return model

    def load_active_model(self):
        a = self.active_version()
        if not a:
            raise ModelStorageError("Belum ada model ACTIVE — jalankan `python main.py setup` atau `train`")
        return self.load(a["version"]), a

    def promote(self, version: str, note: str = "") -> None:
        with self.db.transaction():
            self.db.conn.cursor().execute(self.db._sql("UPDATE model_versions SET status = 'RETIRED' WHERE status = 'ACTIVE'"))
            self.db.conn.cursor().execute(self.db._sql("UPDATE model_versions SET status = 'ACTIVE', promoted_at = ?, "
                                                       "notes = COALESCE(notes, '') || ? WHERE version = ?"),
                                          (now_utc(), f" PROMOTED {note}".rstrip(), version))
        log.info("Model %s dipromosikan menjadi ACTIVE", version)

    def reject(self, version: str, reason: str) -> None:
        self.db.execute("UPDATE model_versions SET status = 'REJECTED', notes = ? WHERE version = ?",
                        (f"REJECTED: {reason}"[:2000], version))
        log.warning("Model %s DITOLAK: %s", version, reason)

    def _backend_for(self, row: dict) -> "ModelStorage":
        b = row.get("storage_backend")
        if b == self.backend:
            return self
        return {"local": LocalModelStorage, "database": DatabaseModelStorage,
                "supabase": SupabaseModelStorage}[b](self.cfg, self.repo)


class LocalModelStorage(ModelStorage):
    backend = "local"

    def _dir(self) -> Path:
        p = resolve_path(self.cfg, get(self.cfg, "storage.local_model_dir", "models"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _put(self, version, blob):
        p = self._dir() / f"{version}.joblib"
        p.write_bytes(blob)
        return str(p), None

    def _get(self, row):
        p = Path(row["storage_uri"])
        if not p.exists():
            raise ModelStorageError(f"File model lokal hilang: {p} (runner ephemeral? pakai storage database/supabase)")
        return p.read_bytes()


class DatabaseModelStorage(ModelStorage):
    backend = "database"

    def _put(self, version, blob):
        return f"db://model_versions/{version}", blob

    def _get(self, row):
        blob = row.get("artifact")
        if blob is None:
            raise ModelStorageError("Artefak model kosong di database")
        return bytes(blob)


class SupabaseModelStorage(ModelStorage):
    """Supabase Storage REST API. Butuh SUPABASE_URL dan SUPABASE_KEY (service role) di environment."""
    backend = "supabase"

    def __init__(self, cfg, repo):
        super().__init__(cfg, repo)
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.key = os.environ.get("SUPABASE_KEY", "")
        self.bucket = get(cfg, "storage.supabase_bucket", "swing-artifacts")
        if not (self.url and self.key):
            raise ModelStorageError("SUPABASE_URL / SUPABASE_KEY belum diset")

    def _req(self, method, path, data=None, ctype="application/octet-stream"):
        import requests
        h = {"Authorization": f"Bearer {self.key}", "apikey": self.key}
        if data is not None:
            h["Content-Type"], h["x-upsert"] = ctype, "true"
        r = requests.request(method, f"{self.url}/storage/v1/object/{self.bucket}/{path}", headers=h, data=data, timeout=120)
        if r.status_code >= 300:
            raise ModelStorageError(f"Supabase Storage {method} {path}: HTTP {r.status_code}")
        return r

    def _put(self, version, blob):
        path = f"models/{version}.joblib"
        self._req("POST", path, blob)
        return f"supabase://{self.bucket}/{path}", None

    def _get(self, row):
        return self._req("GET", row["storage_uri"].split(f"{self.bucket}/", 1)[-1]).content


def make_model_storage(cfg: dict, repo: Repository) -> ModelStorage:
    b = str(get(cfg, "storage.model_backend", "auto")).lower()
    if b == "auto":
        b = "database" if repo.db.dialect == "postgres" else "local"
    cls = {"local": LocalModelStorage, "database": DatabaseModelStorage, "supabase": SupabaseModelStorage}.get(b)
    if cls is None:
        raise ModelStorageError(f"storage.model_backend tidak dikenal: {b}")
    return cls(cfg, repo)
