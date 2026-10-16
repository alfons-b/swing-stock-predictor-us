"""Notifikasi opsional setelah daily run. Aktif hanya bila secret tersedia:
TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, atau NOTIFY_WEBHOOK_URL (Slack/Discord-compatible JSON {"text": ...}).
Isi pesan hanya ringkasan publik — tidak pernah memuat credential.
"""
from __future__ import annotations

import os


def _text(summary: dict) -> str:
    lines = [f"Swing scan {summary['date']} — {summary['headline']}",
             f"Regime: {summary['market_regime']} | Data: {summary['data_freshness']['status']} | Model: {summary['model_version']}"]
    for r in summary["recommendations"][:10]:
        lines.append(f"{r['rank']}. {r['ticker']} {r['setup']} skor {r['score']:.0f} P(bull) {r['probability_bullish']:.0%} "
                     f"entry ${r['entry_low']:,.2f}-{r['entry_high']:,.2f} SL ${r['stop_loss']:,.2f} TP1 ${r['take_profit_1']:,.2f}")
    if summary.get("data_warning"):
        lines.append("⚠ " + summary["data_warning"])
    lines.append("Bukan nasihat investasi.")
    return "\n".join(lines)


def notify_daily(summary: dict) -> bool:
    import requests
    text = _text(summary)
    sent = False
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id": chat, "text": text}, timeout=20)
        sent = True
    hook = os.environ.get("NOTIFY_WEBHOOK_URL")
    if hook:
        requests.post(hook, json={"text": text, "content": text}, timeout=20)
        sent = True
    return sent
