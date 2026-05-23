"""SQLite/PostgreSQL veritabanı katmanı — tüm collector çıktılarını kaydeder."""

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import DATABASE_URL


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


# ── Tablolar ──────────────────────────────────────────────────────────────────

class MarketData(Base):
    """OHLCV ve ticker verisi."""
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False)
    symbol = Column(String(30), nullable=False, index=True)
    data_type = Column(String(30), nullable=False)
    interval = Column(String(10))
    open_time = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    extra = Column(Text)  # JSON string — ek alanlar
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))


class NewsItem(Base):
    """Haber ve KAP açıklamaları."""
    __tablename__ = "news_items"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String(64), unique=True, nullable=False, index=True)
    source = Column(String(100), nullable=False)
    symbol = Column(String(30), index=True)
    title = Column(Text, nullable=False)
    url = Column(Text)
    summary = Column(Text)
    category = Column(String(100))
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_processed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc))


class AgentDecision(Base):
    """Ajan kararları ve gerekçeleri."""
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(30), index=True)
    signal = Column(String(20))         # bullish / bearish / neutral
    confidence = Column(Float)
    reasoning = Column(Text)
    key_points = Column(Text)           # JSON list
    raw_output = Column(Text)           # Tam LLM yanıtı
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(tz=timezone.utc), index=True)


class Trade(Base):
    """Paper ve live trade kayıtları."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String(10), nullable=False)     # paper | live
    symbol = Column(String(30), nullable=False, index=True)
    side = Column(String(10), nullable=False)      # buy | sell
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    pnl = Column(Float)
    status = Column(String(20), default="open")   # open | closed | cancelled
    decision_id = Column(Integer)
    opened_at = Column(DateTime(timezone=True), nullable=False, index=True)
    closed_at = Column(DateTime(timezone=True))
    notes = Column(Text)


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def init_db() -> None:
    """Tüm tabloları oluşturur (yoksa)."""
    Base.metadata.create_all(bind=engine)
    logger.info("Veritabanı başlatıldı: {}", DATABASE_URL)


def get_session() -> Session:
    """Yeni bir DB session döner. Kullandıktan sonra kapat."""
    return SessionLocal()


def save_collector_output(output: dict) -> None:
    """
    Herhangi bir collector çıktısını uygun tabloya kaydeder.

    Args:
        output: Standart CollectorOutput formatında dict
    """
    data_type = output.get("data_type", "")
    if data_type == "ohlcv":
        _save_ohlcv(output)
    elif data_type == "news":
        _save_news(output)
    elif data_type == "macro":
        # Makro veri OHLCV formatında kaydedilir (close = değer)
        _save_ohlcv(output)
    elif data_type in ("ticker", "fundamental", "spread", "order_book"):
        _save_as_extra(output)
    else:
        logger.warning("Bilinmeyen data_type: {}", data_type)


def _save_ohlcv(output: dict) -> None:
    candles: list[dict] = output["payload"].get("candles", [])
    interval = output["payload"].get("interval") or output["payload"].get("granularity")
    saved = 0

    with get_session() as session:
        for candle in candles:
            open_time_str = candle.get("open_time", "")
            try:
                open_time = datetime.fromisoformat(open_time_str)
            except ValueError:
                logger.warning("Geçersiz tarih formatı: {}", open_time_str)
                continue

            # Aynı sembol+interval+open_time varsa atla
            exists = session.query(MarketData).filter_by(
                symbol=output["symbol"],
                interval=interval,
                open_time=open_time,
            ).first()
            if exists:
                continue

            row = MarketData(
                source=output["source"],
                symbol=output["symbol"],
                data_type="ohlcv",
                interval=interval,
                open_time=open_time,
                open=candle.get("open"),
                high=candle.get("high"),
                low=candle.get("low"),
                close=candle.get("close"),
                volume=candle.get("volume"),
            )
            session.add(row)
            saved += 1

        session.commit()

    logger.debug("{} yeni mum kaydedildi — {} {}", saved, output["symbol"], interval)


def _save_news(output: dict) -> None:
    payload = output["payload"]
    uid = payload.get("uid", "")

    with get_session() as session:
        exists = session.query(NewsItem).filter_by(uid=uid).first()
        if exists:
            return

        try:
            published_at = datetime.fromisoformat(output["timestamp"])
        except ValueError:
            published_at = datetime.now(tz=timezone.utc)

        row = NewsItem(
            uid=uid,
            source=output["source"],
            symbol=output.get("symbol"),
            title=payload.get("title", ""),
            url=payload.get("url", ""),
            summary=payload.get("summary", ""),
            category=payload.get("category"),
            published_at=published_at,
        )
        session.add(row)
        session.commit()

    logger.debug("Haber kaydedildi: {}", payload.get("title", "")[:60])


def _save_as_extra(output: dict) -> None:
    """ticker/fundamental gibi yapıları JSON olarak extra alanına kaydeder."""
    with get_session() as session:
        row = MarketData(
            source=output["source"],
            symbol=output["symbol"],
            data_type=output["data_type"],
            open_time=datetime.now(tz=timezone.utc),
            extra=json.dumps(output["payload"], ensure_ascii=False),
        )
        session.add(row)
        session.commit()


def save_agent_decision(
    agent_name: str,
    symbol: str | None,
    signal: str,
    confidence: float,
    reasoning: str,
    key_points: list[str],
    raw_output: str = "",
) -> int:
    """Ajan kararını kaydeder ve yeni satırın id'sini döner."""
    with get_session() as session:
        row = AgentDecision(
            agent_name=agent_name,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_points=json.dumps(key_points, ensure_ascii=False),
            raw_output=raw_output,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def save_trade(trade_data: dict) -> int:
    """Trade kaydeder ve id döner."""
    with get_session() as session:
        row = Trade(**trade_data)
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def get_recent_news(limit: int = 50, symbol: str | None = None) -> list[dict]:
    """Son haberleri döner."""
    with get_session() as session:
        q = session.query(NewsItem).order_by(NewsItem.published_at.desc())
        if symbol:
            q = q.filter(NewsItem.symbol == symbol)
        rows = q.limit(limit).all()
        return [
            {
                "id": r.id,
                "source": r.source,
                "symbol": r.symbol,
                "title": r.title,
                "url": r.url,
                "summary": r.summary,
                "published_at": r.published_at.isoformat() if r.published_at else None,
            }
            for r in rows
        ]


def get_ohlcv(
    symbol: str,
    interval: str,
    limit: int = 500,
) -> list[dict]:
    """DB'den OHLCV verisi çeker (en yeni başta)."""
    with get_session() as session:
        rows = (
            session.query(MarketData)
            .filter_by(symbol=symbol, interval=interval, data_type="ohlcv")
            .order_by(MarketData.open_time.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "open_time": r.open_time.isoformat() if r.open_time else None,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in reversed(rows)  # kronolojik sıra
        ]
