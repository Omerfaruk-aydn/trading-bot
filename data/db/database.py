"""SQLite veritabanı — işlem, karar ve portföy geçmişi.

Tablolar:
  trades             — alış/satış kayıtları
  decisions          — her sembol için model kararları
  portfolio_snapshots — tarama başı portföy anlık görüntüsü
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger


class TradeDB:
    """Thread-safe SQLite veritabanı yöneticisi."""

    def __init__(self, db_path: str = "logs/trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        logger.info("TradeDB hazır: {}", db_path)

    # ── Şema ─────────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            symbol      TEXT    NOT NULL,
            action      TEXT    NOT NULL,   -- buy | sell
            shares      REAL    NOT NULL,
            price       REAL    NOT NULL,
            pnl         REAL    DEFAULT 0,
            reason      TEXT    DEFAULT '',
            mode        TEXT    DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            symbol      TEXT    NOT NULL,
            action      TEXT    NOT NULL,   -- buy | sell | hold
            confidence  REAL    NOT NULL,
            reason      TEXT    DEFAULT '',
            price       REAL    DEFAULT 0,
            rsi         REAL    DEFAULT 0,
            macd        REAL    DEFAULT 0,
            sentiment   REAL    DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            total_value REAL    NOT NULL,
            cash        REAL    NOT NULL,
            pnl         REAL    NOT NULL,
            positions   INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol   ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_action   ON trades(action);
        CREATE INDEX IF NOT EXISTS idx_decisions_sym   ON decisions(symbol);
        """)
        self._conn.commit()

    # ── Yazma ────────────────────────────────────────────────────────────────

    def log_trade(
        self,
        symbol: str,
        action: str,
        shares: float,
        price: float,
        pnl: float = 0.0,
        reason: str = "",
        mode: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO trades (timestamp,symbol,action,shares,price,pnl,reason,mode) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), symbol, action, shares, price, pnl, reason, mode),
            )
            self._conn.commit()

    def log_decision(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reason: str = "",
        price: float = 0.0,
        rsi: float = 0.0,
        macd: float = 0.0,
        sentiment: float = 0.0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO decisions (timestamp,symbol,action,confidence,reason,price,rsi,macd,sentiment) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), symbol, action, confidence, reason, price, rsi, macd, sentiment),
            )
            self._conn.commit()

    def log_snapshot(self, total_value: float, cash: float, pnl: float, positions: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO portfolio_snapshots (timestamp,total_value,cash,pnl,positions) "
                "VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(), total_value, cash, pnl, positions),
            )
            self._conn.commit()

    # ── Okuma / Analiz ───────────────────────────────────────────────────────

    def get_win_stats(self, symbol: str | None = None, last_n: int = 50) -> dict:
        """
        Son N satış işlemine göre kazanma oranı ve ortalama P&L döndürür.
        Kelly Criterion hesabı için kullanılır.
        """
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    "SELECT pnl FROM trades WHERE action='sell' AND symbol=? "
                    "ORDER BY id DESC LIMIT ?",
                    (symbol, last_n),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT pnl FROM trades WHERE action='sell' "
                    "ORDER BY id DESC LIMIT ?",
                    (last_n,),
                ).fetchall()

        if not rows:
            return {"win_rate": 0.5, "loss_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0, "count": 0}

        pnls = [r["pnl"] for r in rows]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        avg_win  = sum(wins)  / len(wins)  if wins  else 0.01
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.01

        return {
            "win_rate":  len(wins) / len(pnls),
            "loss_rate": len(losses) / len(pnls),
            "avg_win":   avg_win,
            "avg_loss":  avg_loss,
            "count":     len(pnls),
        }

    def kelly_fraction(self, symbol: str | None = None, last_n: int = 50, half_kelly: bool = True) -> float:
        """
        Kelly Criterion ile optimum pozisyon büyüklüğü (0.0 – 1.0).
        Yeterli geçmiş yoksa 0.0 döner (varsayılan max_pos kullanılır).

        f* = (b*p - q) / b
          b = avg_win / avg_loss (ödül/risk oranı)
          p = kazanma oranı
          q = 1 - p
        """
        stats = self.get_win_stats(symbol, last_n)
        if stats["count"] < 10:
            return 0.0  # Yeterli veri yok

        p = stats["win_rate"]
        q = stats["loss_rate"]
        b = stats["avg_win"] / max(stats["avg_loss"], 0.01)

        f = (b * p - q) / b
        f = max(0.0, min(f, 1.0))  # 0-1 aralığına sıkıştır

        return f / 2 if half_kelly else f  # Half-Kelly daha güvenli

    def recent_trades(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def symbol_stats(self, symbol: str) -> dict:
        """Bir sembol için toplam işlem istatistikleri."""
        with self._lock:
            sells = self._conn.execute(
                "SELECT pnl FROM trades WHERE action='sell' AND symbol=?", (symbol,)
            ).fetchall()
        pnls = [r["pnl"] for r in sells]
        total_pnl = sum(pnls)
        win_count = sum(1 for p in pnls if p > 0)
        return {
            "symbol": symbol,
            "total_trades": len(pnls),
            "win_count": win_count,
            "loss_count": len(pnls) - win_count,
            "win_rate": win_count / len(pnls) if pnls else 0,
            "total_pnl": total_pnl,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
