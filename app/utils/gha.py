"""Annotation GitHub Actions (::notice / ::warning) — ringkasan hasil terlihat di halaman run TANPA membuka log,
dan bisa dibaca lewat API (check-runs/{id}/annotations). Di luar Actions tidak mencetak apa pun.

Jangan pernah memasukkan secret ke sini (annotation bersifat publik di repo publik)."""
from __future__ import annotations

import os


def _esc(s: str) -> str:
    return str(s).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotate(title: str, message: str, level: str = "notice") -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    level = level if level in ("notice", "warning", "error") else "notice"
    t = _esc(title).replace(",", "%2C").replace("::", "%3A%3A")
    print(f"::{level} title={t}::{_esc(message)[:4000]}", flush=True)
