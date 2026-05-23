"""DB'de kayıtlı hisse verilerinden stocks checkpoint'i yeniden oluşturur."""
import json
from pathlib import Path
from data.storage import init_db, get_session, MarketData

init_db()

CHECKPOINT_FILE = Path("logs/stocks_checkpoint.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

from config.symbols import (
    BIST_SYMBOLS, SP500_SYMBOLS, EUROPE_SYMBOLS, ASIA_SYMBOLS,
    FOREX_PAIRS, COMMODITY_SYMBOLS, INDEX_SYMBOLS,
    TIMEFRAMES_STOCK, TIMEFRAMES_FOREX,
)

groups = {
    "bist":      (BIST_SYMBOLS,     TIMEFRAMES_STOCK),
    "sp500":     (SP500_SYMBOLS,    TIMEFRAMES_STOCK),
    "europe":    (EUROPE_SYMBOLS,   TIMEFRAMES_STOCK),
    "asia":      (ASIA_SYMBOLS,     TIMEFRAMES_STOCK),
    "forex":     (FOREX_PAIRS,      TIMEFRAMES_FOREX),
    "commodity": (COMMODITY_SYMBOLS,TIMEFRAMES_STOCK),
    "index":     (INDEX_SYMBOLS,    TIMEFRAMES_STOCK),
}

done = []
with get_session() as session:
    for group_name, (symbols, timeframes) in groups.items():
        for symbol in symbols:
            for interval in timeframes:
                exists = session.query(MarketData).filter_by(
                    symbol=symbol, interval=interval
                ).first()
                if exists:
                    key = f"yf:{group_name}:{symbol}:{interval}"
                    done.append(key)

CHECKPOINT_FILE.write_text(
    json.dumps({"done": done}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Checkpoint yeniden oluşturuldu: {len(done)} tamamlanmış işlem")
