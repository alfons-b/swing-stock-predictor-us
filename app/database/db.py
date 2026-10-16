"""Database abstraction — SQLite (dev/test/CI) dan PostgreSQL (production: Aiven, lihat docs/DATABASE_DECISION.md).

Keputusan desain: lapisan tipis di atas DB-API (sqlite3 / psycopg 3), bukan ORM.
- Semua SQL ditulis dengan placeholder `?`; dikonversi ke `%s` untuk Postgres.
- UPSERT memakai `INSERT ... ON CONFLICT (...) DO UPDATE`, didukung keduanya → idempoten.
- Skema didefinisikan sekali (schema.py) lalu di-render per dialek.
"""
from __future__ import annotations

import datetime
import decimal
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from app.utils.logging_utils import get_logger

log = get_logger(__name__)


class DatabaseError(RuntimeError):
    pass


class DatabaseAuthError(DatabaseError):
    """Kredensial/nama database salah. Tidak bersifat sementara → tidak dicoba ulang."""


# Pesan server yang berarti DATABASE_URL salah isi (bukan gangguan jaringan) — percobaan ulang tidak akan menolong.
_AUTH_ERRORS = ("password authentication failed", "no password supplied", "role \"", "does not exist",
                "no pg_hba.conf entry", "invalid dsn", "missing \"=\"")


def _py(v):
    """Konversi nilai numpy/pandas ke tipe Python yang diterima driver DB."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else (v.strftime("%Y-%m-%d") if v == v.normalize() else v.isoformat())
    if v is pd.NaT:
        return None
    if isinstance(v, datetime.datetime):  # PostgreSQL mengembalikan objek tanggal; SQLite mengembalikan teks
        return v.isoformat()
    if isinstance(v, datetime.date):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


class Database:
    dialect = "base"
    placeholder = "?"

    def __init__(self, url: str):
        self.url = url
        self._conn = None
        self._in_tx = False          # True di dalam `with db.transaction()` (atomik: tidak ada commit/retry otomatis)

    # ------------------------------------------------------------------ factory
    @staticmethod
    def from_url(url: str, root: str | Path | None = None) -> "Database":
        if not url:
            raise DatabaseError("DATABASE_URL kosong")
        if url.startswith("sqlite"):
            return SQLiteDatabase(url, root)
        if url.startswith(("postgres://", "postgresql://")):
            return PostgresDatabase(url)
        raise DatabaseError(f"Skema URL database tidak didukung: {url.split(':')[0]}")

    # ------------------------------------------------------------------ core
    def connect(self):
        raise NotImplementedError

    @property
    def conn(self):
        if self._conn is None:
            self._conn = self.connect()
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _sql(self, sql: str) -> str:
        """Placeholder `?` → `%s` untuk psycopg. `%` literal (mis. LIKE 'wf-test-%') di-escape menjadi `%%`, karena
        psycopg membaca setiap `%` sebagai penanda parameter (ProgrammingError — insiden dashboard 13 Okt 2026)."""
        if self.placeholder == "?":
            return sql
        return sql.replace("%", "%%").replace("?", self.placeholder)

    def _recover(self) -> None:
        """Setelah statement gagal: PostgreSQL menandai transaksi 'aborted' dan MENOLAK semua query berikutnya
        (InFailedSqlTransaction) sampai rollback. Koneksi yang mati (jaringan/pooler menutup) dibuang agar
        panggilan berikutnya membuka koneksi baru — penting untuk koneksi berumur panjang (dashboard)."""
        if self._conn is None:
            return
        try:
            self._conn.rollback()
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            return
        if getattr(self._conn, "closed", False):
            self._conn = None

    def _run(self, fn, read: bool = False):
        """Jalankan satu operasi. Dua aturan untuk job panjang di PostgreSQL terkelola (Neon):

        1. Baca di luar transaksi eksplisit → transaksi implisit langsung ditutup (commit). Tanpa ini sesi tertinggal
           'idle in transaction' selama komputasi CPU berikutnya, dan server memutusnya
           (IdleInTransactionSessionTimeout — insiden Initial Setup 12 Okt 2026).
        2. Koneksi diputus server (timeout, compute Neon tidur/restart, jaringan) → buka koneksi baru dan ulangi SEKALI.
           Aman karena di luar transaksi eksplisit setiap operasi berdiri sendiri: kegagalan sebelum commit berarti
           tidak ada yang tersimpan, dan penulisan di proyek ini idempoten (UPSERT/UPDATE/DELETE berkunci).
           Di dalam `transaction()` tidak ada retry: error dilempar agar atomisitas tidak dikorbankan.
        """
        attempts = 1 if self._in_tx else 2
        for attempt in range(attempts):
            try:
                out = fn(self.conn.cursor())
                if read and not self._in_tx:
                    self.conn.commit()
                return out
            except Exception as e:
                if self._in_tx or isinstance(e, DatabaseAuthError):
                    raise                               # transaction() yang rollback / kredensial salah: jangan ulang
                self._recover()
                if self._conn is not None or attempt + 1 >= attempts:
                    raise                               # error SQL biasa, atau sudah dicoba ulang
                log.warning("Koneksi database terputus (%s) — membuka koneksi baru dan mengulang sekali",
                            type(e).__name__)

    def execute(self, sql: str, params: tuple | list = ()) -> int:
        def f(cur):
            cur.execute(self._sql(sql), [_py(p) for p in params])
            n = cur.rowcount
            self.conn.commit()
            return n
        return self._run(f)

    def executescript(self, statements: list[str]) -> None:
        def f(cur):
            for st in statements:
                cur.execute(st)
            self.conn.commit()
        self._run(f)

    def query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        def f(cur):
            cur.execute(self._sql(sql), [_py(p) for p in params])
            return cur.fetchall()
        return self._run(f, read=True)

    def query_df(self, sql: str, params: tuple | list = (), parse_dates: list[str] | None = None) -> pd.DataFrame:
        def f(cur):
            cur.execute(self._sql(sql), [_py(p) for p in params])
            return [d[0] for d in cur.description], cur.fetchall()
        cols, rows = self._run(f, read=True)
        df = pd.DataFrame(rows, columns=cols)
        for c in parse_dates or []:
            if c in df:
                df[c] = pd.to_datetime(df[c])
        return df

    def scalar(self, sql: str, params: tuple | list = ()):
        rows = self.query(sql, params)
        return rows[0][0] if rows else None

    def executemany(self, sql: str, rows: list) -> None:
        def f(cur):
            cur.executemany(self._sql(sql), rows)
            self.conn.commit()
        self._run(f)

    @contextmanager
    def transaction(self):
        """Blok atomik: commit di akhir, rollback bila ada error. Tanpa commit/retry otomatis di dalamnya."""
        if self._in_tx:                                 # bersarang → ikut transaksi luar
            yield self
            return
        self._in_tx = True
        try:
            yield self
            self.conn.commit()
        except Exception:
            if self._conn is not None:                  # koneksi mati → transaksi sudah batal di server
                try:
                    self._conn.rollback()
                except Exception:
                    self._recover()
            raise
        finally:
            self._in_tx = False

    # ------------------------------------------------------------------ upsert
    def upsert(self, table: str, df: pd.DataFrame | list[dict], keys: list[str], update: list[str] | None = None,
               batch_size: int = 2000, count: bool = True) -> tuple[int, int]:
        """Idempotent UPSERT. Return (inserted, updated).

        Menjalankan dua kali dengan data sama → tidak ada duplikat (unique constraint pada `keys`).
        """
        if isinstance(df, list):
            df = pd.DataFrame(df)
        if df is None or len(df) == 0:
            return 0, 0
        df = df.drop_duplicates(subset=keys, keep="last")
        cols = list(df.columns)
        update = [c for c in (update if update is not None else cols) if c not in keys]
        ph = ", ".join([self.placeholder] * len(cols))
        conflict = ", ".join(keys)
        if update:
            set_clause = ", ".join(f"{c} = excluded.{c}" for c in update)
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph}) ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}"
        else:
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph}) ON CONFLICT ({conflict}) DO NOTHING"
        rows = [tuple(_py(v) for v in r) for r in df.itertuples(index=False, name=None)]

        def f(cur):
            # Hitung + tulis + commit dalam satu unit: bila koneksi putus di tengah, tidak ada yang tersimpan dan
            # _run mengulang seluruhnya di koneksi baru (ON CONFLICT → aman diulang).
            existing = self._count_existing(table, df, keys) if count else 0
            for i in range(0, len(rows), batch_size):
                cur.executemany(sql, rows[i:i + batch_size])
            if not self._in_tx:
                self.conn.commit()
            return existing
        existing = self._run(f)
        return len(rows) - existing, existing if update else 0

    def _count_existing(self, table: str, df: pd.DataFrame, keys: list[str]) -> int:
        """Hitung baris yang sudah ada (untuk statistik inserted vs updated)."""
        k0 = keys[0]
        vals = [_py(v) for v in pd.unique(df[k0])]
        total = 0
        want = set(tuple(_py(v) for v in r) for r in df[keys].itertuples(index=False, name=None))
        extra, params = "", []
        if len(keys) > 1 and keys[1] in ("date", "prediction_date", "signal_date"):
            extra = f" AND {keys[1]} >= ? AND {keys[1]} <= ?"
            col = df[keys[1]]
            params = [_py(col.min()), _py(col.max())]
        for i in range(0, len(vals), 500):
            chunk = vals[i:i + 500]
            ph = ", ".join([self.placeholder] * len(chunk))
            sql = f"SELECT {', '.join(keys)} FROM {table} WHERE {k0} IN ({ph}){extra.replace('?', self.placeholder)}"
            cur = self.conn.cursor()
            cur.execute(sql, chunk + params)
            for r in cur.fetchall():
                if tuple(_py(v) if not isinstance(v, (bytes, memoryview)) else v for v in r) in want:
                    total += 1
        return total


class SQLiteDatabase(Database):
    dialect = "sqlite"
    placeholder = "?"
    provider = "sqlite"

    def __init__(self, url: str, root=None):
        super().__init__(url)
        path = url.split("sqlite:///", 1)[-1] if "sqlite:///" in url else url.split("sqlite://", 1)[-1]
        if path in ("", ":memory:"):
            self.path = ":memory:"
        else:
            p = Path(path)
            if not p.is_absolute() and root is not None:
                p = Path(root) / p
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(p)

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL") if self.path != ":memory:" else None
        return conn

    def describe(self) -> str:
        return f"sqlite:{self.path}"


class PostgresDatabase(Database):
    dialect = "postgres"
    placeholder = "%s"

    def __init__(self, url: str):
        from app.database.providers import normalize_url, profile_for
        super().__init__(normalize_url(url))
        self.profile = profile_for(url)

    @property
    def provider(self) -> str:
        return self.profile.name

    def connect(self):
        try:
            import psycopg
        except ImportError as e:  # pragma: no cover
            raise DatabaseError("psycopg belum terpasang: pip install 'psycopg[binary]'") from e
        last = None
        # Neon scale-to-zero: compute yang suspend butuh beberapa detik untuk bangun; pooler terkelola kadang
        # menolak koneksi sesaat. Backoff 1,2,4,8,15 s (dibatasi) — total < 1 menit pada kasus terburuk.
        for attempt in range(self.profile.connect_attempts):
            try:
                return psycopg.connect(self.url, connect_timeout=self.profile.connect_timeout_s, prepare_threshold=None)
            except Exception as e:  # pragma: no cover - jaringan
                last = e
                msg = str(e).lower()
                if any(k in msg for k in _AUTH_ERRORS):
                    raise DatabaseAuthError(
                        f"Login ke PostgreSQL ({self.provider}) DITOLAK: user/password/nama database di DATABASE_URL "
                        "salah. Salin ulang Service URI lengkap dari console (password asli, bukan tanda bintang), "
                        "lalu perbarui secret DATABASE_URL. Tidak dicoba ulang karena ini bukan gangguan jaringan.") from e
                if attempt + 1 < self.profile.connect_attempts:
                    time.sleep(min(2 ** attempt, 15))
        # Pesan error driver bisa memuat host; tidak pernah memuat password (psycopg menyamarkannya).
        raise DatabaseError(f"Gagal konek ke PostgreSQL ({self.provider}) setelah {self.profile.connect_attempts} "
                            f"percobaan: {type(last).__name__}") from last

    def describe(self) -> str:
        u = urlparse(self.url)
        return f"postgres[{self.provider}]:{u.hostname}:{u.port or 5432}{u.path}"  # tanpa user/password
