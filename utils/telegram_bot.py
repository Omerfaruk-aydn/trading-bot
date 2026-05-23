"""Telegram bildirim sistemi — alış/satış/özet bildirimleri."""

from __future__ import annotations

import os
import threading
import requests
from loguru import logger


class TelegramBot:
    """Telegram Bot API ile bildirim gönderir. Token yoksa sessizce devre dışı kalır."""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._lock = threading.Lock()

        if self.enabled:
            logger.info("Telegram bildirimleri aktif (chat_id: {})", self.chat_id)
        else:
            logger.info("Telegram devre dışı — TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID eksik.")

    # ── İç gönderici ─────────────────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock:
                r = requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=8,
                )
            if not r.ok:
                logger.warning("Telegram hata {}: {}", r.status_code, r.text[:100])
            return r.ok
        except Exception as e:
            logger.warning("Telegram gönderilemedi: {}", e)
            return False

    def _send_async(self, text: str) -> None:
        """Arka planda gönder — ana döngüyü bloklamasın."""
        threading.Thread(target=self._send, args=(text,), daemon=True).start()

    # ── Bildirim şablonları ───────────────────────────────────────────────────

    def buy_alert(
        self,
        symbol: str,
        shares: float,
        price: float,
        confidence: float,
        reason: str,
        currency: str = "TL",
        kelly_pct: float | None = None,
    ) -> None:
        kelly_line = f"\n📐 Kelly: %{kelly_pct:.1f}" if kelly_pct is not None else ""
        text = (
            f"🟢 <b>ALIŞ</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📌 Sembol: <code>{symbol}</code>\n"
            f"📦 Adet: {shares:.4f} @ <b>{price:.2f} {currency}</b>\n"
            f"🎯 Güven: %{confidence*100:.0f}{kelly_line}\n"
            f"💬 {reason[:120]}"
        )
        self._send_async(text)

    def sell_alert(
        self,
        symbol: str,
        shares: float,
        price: float,
        pnl: float,
        reason: str,
        currency: str = "TL",
    ) -> None:
        emoji = "💰" if pnl >= 0 else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""
        text = (
            f"{emoji} <b>SATIŞ</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📌 Sembol: <code>{symbol}</code>\n"
            f"📦 Adet: {shares:.4f} @ <b>{price:.2f} {currency}</b>\n"
            f"💵 P&L: <b>{pnl_sign}{pnl:,.0f} TL</b>\n"
            f"💬 {reason[:120]}"
        )
        self._send_async(text)

    def stop_alert(self, symbol: str, price: float, pnl: float, stop_type: str = "Stop-loss") -> None:
        text = (
            f"⛔ <b>{stop_type} Tetiklendi</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📌 Sembol: <code>{symbol}</code>\n"
            f"💲 Fiyat: {price:.2f}\n"
            f"💵 P&L: <b>{pnl:+,.0f} TL</b>"
        )
        self._send_async(text)

    def target_reached(self, total: float, pnl: float, pnl_pct: float) -> None:
        text = (
            f"🏆 <b>HEDEF ULAŞILDI!</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 Toplam: <b>{total:,.0f} TL</b>\n"
            f"📈 P&L: <b>+{pnl:,.0f} TL (%{pnl_pct:.2f})</b>"
        )
        self._send_async(text)

    def daily_summary(
        self,
        total: float,
        initial: float,
        pnl: float,
        pnl_pct: float,
        target: float,
        win_trades: int,
        loss_trades: int,
        open_positions: int,
    ) -> None:
        progress = min(total / target * 100, 100)
        emoji = "📈" if pnl >= 0 else "📉"
        text = (
            f"{emoji} <b>GÜNLÜK ÖZET</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼 Portföy: <b>{total:,.0f} TL</b>\n"
            f"💵 P&L: <b>{pnl:+,.0f} TL (%{pnl_pct:+.2f})</b>\n"
            f"🎯 Hedefe İlerleme: %{progress:.1f}\n"
            f"✅ Kazanan: {win_trades} | ❌ Kaybeden: {loss_trades}\n"
            f"📂 Açık Pozisyon: {open_positions}"
        )
        self._send_async(text)

    def warning(self, message: str) -> None:
        self._send_async(f"⚠️ <b>UYARI</b>\n{message}")

    def info(self, message: str) -> None:
        self._send_async(f"ℹ️ {message}")

    def economic_event(self, event: str, impact: str, time: str) -> None:
        emoji = "🔴" if impact == "high" else "🟡"
        self._send_async(
            f"{emoji} <b>Ekonomik Olay</b>\n"
            f"📅 {time}\n"
            f"📌 {event}\n"
            f"⚡ Etki: {impact.upper()}"
        )
