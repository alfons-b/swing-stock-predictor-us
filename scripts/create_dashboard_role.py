"""Buat role READ-ONLY `dashboard_ro` di database Aiven untuk dashboard (jalankan sekali, dari laptop).

Pakai:
    # PowerShell
    $env:DATABASE_URL = "postgres://avnadmin:...@pg-....aivencloud.com:12345/defaultdb?sslmode=require"
    python scripts/create_dashboard_role.py
    # macOS / Linux
    DATABASE_URL='postgres://avnadmin:...' python scripts/create_dashboard_role.py

Skrip meminta password baru untuk `dashboard_ro` (tidak ditampilkan saat diketik), menjalankan GRANT SELECT, lalu
mencetak connection string untuk Secrets Streamlit/Render. Jalankan SETELAH workflow Initial Setup membuat tabel.
Aman diulang: role yang sudah ada hanya diganti password-nya.
"""
from __future__ import annotations

import getpass
import os
import sys
from urllib.parse import quote, urlparse, urlunparse

ROLE = "dashboard_ro"


def main() -> int:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url.startswith(("postgres://", "postgresql://")):
        print("Set DATABASE_URL ke Service URI Aiven (user avnadmin) lebih dulu.", file=sys.stderr)
        return 2
    try:
        import psycopg
        from psycopg import sql
    except ImportError:
        print("pip install -r requirements.txt dulu (butuh psycopg).", file=sys.stderr)
        return 2
    pw = os.environ.get("DASHBOARD_RO_PASSWORD") or getpass.getpass(f"Password baru untuk {ROLE}: ")
    if len(pw) < 12:
        print("Password minimal 12 karakter.", file=sys.stderr)
        return 2
    u = urlparse(url)
    db = (u.path or "/defaultdb").lstrip("/") or "defaultdb"
    owner = u.username or "avnadmin"
    with psycopg.connect(url, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,)).fetchone()
        verb = "ALTER" if exists else "CREATE"
        conn.execute(sql.SQL(verb + " ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(ROLE), sql.Literal(pw)))
        for stmt in (
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(db), sql.Identifier(ROLE)),
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(ROLE)),
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(ROLE)),
            sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(
                sql.Identifier(owner), sql.Identifier(ROLE)),
        ):
            conn.execute(stmt)
        n = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'").fetchone()[0]
    host = (u.hostname or "") + (f":{u.port}" if u.port else "")
    ro = urlunparse(u._replace(netloc=f"{ROLE}:{quote(pw, safe='')}@{host}"))
    print(f"Role {ROLE} {'diperbarui' if exists else 'dibuat'}; SELECT diberikan pada {n} tabel.")
    if n == 0:
        print("PERINGATAN: belum ada tabel — jalankan workflow Initial Setup dulu, lalu ulangi skrip ini.")
    print("\nTempel di Secrets dashboard (Streamlit Cloud → app → Settings → Secrets):\n")
    print(f'DATABASE_URL = "{ro}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
