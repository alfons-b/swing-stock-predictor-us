"""Profil provider PostgreSQL terkelola (docs/DATABASE_DECISION.md).

Kode aplikasi tidak bergantung pada vendor: hanya `DATABASE_URL` PostgreSQL standar. Profil ini dipakai untuk
hal-hal yang memang berbeda antar-vendor dan berdampak operasional:
  - batas ukuran paket gratis → health check WARN/FAIL dan retensi otomatis sebelum penulisan diblokir;
  - SSL wajib untuk host publik (Neon/Aiven menolak koneksi tanpa TLS);
  - retry koneksi lebih panjang untuk layanan yang bisa sedang tidur/dinyalakan ulang.

Versi US memakai **Aiven for PostgreSQL Free** (Neon & Supabase dipakai proyek lain pemilik repo dan diblokir di
produksi oleh app/pipeline/context.guard_production_database).

Angka batas diambil dari halaman harga/dokumentasi resmi (dicek 2026-10-16) dan SENGAJA dibulatkan ke bawah
karena metode pengukuran vendor tidak identik dengan `pg_database_size`. Override selalu lewat `DB_SIZE_LIMIT_MB`.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "postgres", "db", "host.docker.internal"}


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    free_size_limit_mb: float     # batas efektif yang dipakai health check (dibulatkan ke bawah)
    connect_attempts: int
    connect_timeout_s: int
    requires_ssl: bool
    on_limit: str                 # apa yang terjadi saat batas terlampaui (ditampilkan di health check)
    recommended: bool             # sesuai keputusan di docs/DATABASE_DECISION.md


PROFILES: dict[str, ProviderProfile] = {
    # Aiven Free (DIPILIH): 1 GB disk (termasuk WAL/sistem), 1 CPU, 1 GB RAM, max_connections 20, tanpa batas jam
    # compute. Layanan tanpa aktivitas berkelanjutan bisa di-power-off (ada notifikasi email) → job harian +
    # Database Health Check harian menjaga aktivitas; bila tetap mati, nyalakan ulang di console Aiven.
    "aiven": ProviderProfile("aiven", 800, 6, 30, True,
                             "disk penuh → penulisan gagal; layanan idle bisa di-power-off Aiven (nyalakan ulang di console)", True),
    # Neon Free: dipakai proyek BEI pemilik repo → diblokir di produksi versi US (ALLOW_NEON_DATABASE untuk override).
    "neon": ProviderProfile("neon", 450, 6, 30, True,
                            "penulisan diblokir sampai ukuran turun atau paket di-upgrade", False),
    # Supabase Free: 500 MB → project READ-ONLY. Dipakai aplikasi lain pemilik repo → diblokir di produksi.
    "supabase": ProviderProfile("supabase", 500, 3, 20, True,
                                "project menjadi READ-ONLY", False),
    "local_postgres": ProviderProfile("local_postgres", 100_000, 3, 10, False, "-", False),
    "postgres": ProviderProfile("postgres", 1000, 4, 20, False,
                                "tidak diketahui — set DB_SIZE_LIMIT_MB sesuai paket Anda", False),
    "sqlite": ProviderProfile("sqlite", 100_000, 1, 30, False, "-", False),
}


def detect_provider(url: str | None) -> str:
    if not url:
        return "sqlite"
    if url.startswith("sqlite"):
        return "sqlite"
    host = (urlparse(url).hostname or "").lower()
    if host.endswith(".neon.tech") or ".neon.tech" in host or host.endswith(".neon.build"):
        return "neon"
    if host.endswith(".aivencloud.com"):
        return "aiven"
    if "supabase.co" in host or "supabase.com" in host:
        return "supabase"
    if host in LOCAL_HOSTS:
        return "local_postgres"
    return "postgres"


def profile_for(url: str | None) -> ProviderProfile:
    return PROFILES[detect_provider(url)]


def resolve_size_limit_mb(value, url: str | None) -> float:
    """`database.size_limit_mb`: angka eksplisit (DB_SIZE_LIMIT_MB) atau 'auto' → batas profil provider."""
    if value is None or str(value).strip().lower() in ("", "auto"):
        return float(profile_for(url).free_size_limit_mb)
    return float(value)


def normalize_url(url: str) -> str:
    """Tambahkan sslmode=require untuk host publik bila belum ada. Tidak pernah menurunkan sslmode yang diset user."""
    if not url.startswith(("postgres://", "postgresql://")):
        return url
    prof = profile_for(url)
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    if "sslmode" not in q and (prof.requires_ssl or prof.name == "postgres"):
        q["sslmode"] = "require"
        return urlunparse(u._replace(query=urlencode(q)))
    return url


def is_pooled_neon(url: str) -> bool:
    """Neon: host dengan akhiran '-pooler' = PgBouncer (transaction mode). Aplikasi ini aman di pooler karena
    prepared statement dimatikan (prepare_threshold=None) dan tidak ada state sesi yang dipakai lintas transaksi."""
    return "-pooler." in (urlparse(url).hostname or "")
