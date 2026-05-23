"""FRED + World Bank Makro Ekonomik Veri Toplama (pandas-datareader).

Kullanım:
    py collect_fred.py

Checkpoint: logs/fred_checkpoint.json
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from data.storage import init_db, save_collector_output
from config.symbols import FRED_SERIES

CHECKPOINT_FILE = Path("logs/fred_checkpoint.json")
CHECKPOINT_FILE.parent.mkdir(exist_ok=True)

START_DATE = "2000-01-01"


def _load_checkpoint() -> list[str]:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8")).get("done", [])
        except Exception:
            pass
    return []


def _save_checkpoint(done: list[str]) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"done": done}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fetch_fred(series_id: str, api_key: str) -> list[dict]:
    """FRED REST API ile tek seri çeker (requests, timeout=20)."""
    import requests
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": START_DATE,
            },
            timeout=20,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
    except Exception as exc:
        logger.warning("FRED API hata: {} — {}", series_id, exc)
        return []

    result = []
    for o in obs:
        try:
            val = float(o["value"])
        except (ValueError, KeyError):
            continue
        result.append({
            "open_time": o["date"] + "T00:00:00+00:00",
            "open": val,
            "high": val,
            "low": val,
            "close": val,
            "volume": 0.0,
        })
    return result


def _fetch_worldbank(indicator: str, description: str) -> list[dict]:
    """World Bank API'den gösterge çeker (key gerektirmez)."""
    try:
        import pandas_datareader.data as web
        df = web.DataReader(indicator, "wb", start=2000, end=2025)
        if df is None or df.empty:
            return []
        df = df.dropna()
        result = []
        for ts, row in df.iterrows():
            val = float(row.iloc[0])
            result.append({
                "open_time": f"{ts}-01-01",
                "open": val,
                "high": val,
                "low": val,
                "close": val,
                "volume": 0.0,
            })
        return result
    except Exception as exc:
        logger.warning("World Bank hata: {} — {}", indicator, exc)
        return []


def main():
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        logger.error("FRED_API_KEY bulunamadı — .env dosyasını kontrol et")
        return

    init_db()

    done = _load_checkpoint()
    remaining = [(sid, desc) for sid, desc in FRED_SERIES.items()
                 if f"fred:{sid}" not in done]

    logger.info("=" * 60)
    logger.info("FRED MAKRO VERİ TOPLAMA (pandas-datareader)")
    logger.info("Toplam: {} | Tamamlanan: {} | Kalan: {}",
                len(FRED_SERIES), len(done), len(remaining))
    logger.info("=" * 60)

    total = len(remaining)
    for i, (series_id, description) in enumerate(remaining, 1):
        key = f"fred:{series_id}"
        logger.info("[{}/{}] {}: {}", i, total, series_id, description)

        data = _fetch_fred(series_id, api_key)

        if not data:
            logger.warning("  → Veri yok, atlanıyor")
            done.append(key)
            _save_checkpoint(done)
            time.sleep(0.3)
            continue

        output = {
            "source": "fred",
            "symbol": series_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "data_type": "macro",
            "payload": {
                "interval": "1mo",
                "description": description,
                "series_id": series_id,
                "candles": data,
            },
        }
        save_collector_output(output)
        done.append(key)
        _save_checkpoint(done)
        logger.info("  → {} veri noktası kaydedildi", len(data))
        time.sleep(0.2)

    logger.info("=" * 60)
    logger.info("FRED TAMAMLANDI — {} seri işlendi", len(FRED_SERIES))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
