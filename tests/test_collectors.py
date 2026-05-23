"""Collector modülleri için temel testler."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


# ── Crypto Collector ──────────────────────────────────────────────────────────

class TestCryptoCollector:
    def test_fetch_ohlcv_invalid_interval(self):
        from data.collectors.crypto_collector import fetch_ohlcv
        with pytest.raises(ValueError, match="Geçersiz interval"):
            fetch_ohlcv("BTCUSDT", interval="3m")

    def test_fetch_ohlcv_uses_cache(self):
        from data.collectors import crypto_collector

        mock_data = {"source": "binance", "symbol": "BTCUSDT", "timestamp": "t",
                     "data_type": "ohlcv", "payload": {"interval": "1h", "candles": []}}

        with patch.object(crypto_collector, "_make_request", return_value=[]) as mock_req:
            # İlk çağrı — ağdan gelir
            with patch.object(crypto_collector, "_is_cached", return_value=False):
                with patch.object(crypto_collector, "_set_cache"):
                    with patch.object(crypto_collector, "_get_cached", return_value=mock_data):
                        pass  # cache miss → request

            # İkinci çağrı — cache'den gelir
            with patch.object(crypto_collector, "_is_cached", return_value=True):
                with patch.object(crypto_collector, "_get_cached", return_value=mock_data):
                    result = crypto_collector.fetch_ohlcv("BTCUSDT", "1h", 10)
                    assert result["symbol"] == "BTCUSDT"
                    mock_req.assert_not_called()


# ── News Collector ────────────────────────────────────────────────────────────

class TestNewsCollector:
    def test_url_hash_unique(self):
        from data.collectors.news_collector import _url_hash
        h1 = _url_hash("https://example.com/news/1")
        h2 = _url_hash("https://example.com/news/2")
        assert h1 != h2

    def test_url_hash_deterministic(self):
        from data.collectors.news_collector import _url_hash
        url = "https://example.com/news/test"
        assert _url_hash(url) == _url_hash(url)

    def test_filter_by_symbols(self):
        from data.collectors.news_collector import filter_by_symbols
        news = [
            {"payload": {"title": "THYAO yeni sefer açıklıyor", "summary": ""}, "symbol": None},
            {"payload": {"title": "Kripto piyasaları yükseliyor", "summary": "BTC rallisi"}, "symbol": None},
        ]
        result = filter_by_symbols(news, ["THYAO.IS"])
        assert len(result) == 1
        assert result[0]["symbol"] == "THYAO.IS"


# ── Storage ───────────────────────────────────────────────────────────────────

class TestStorage:
    def test_init_db_creates_tables(self, tmp_path):
        import os
        os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"

        # settings'i yeniden yükle
        import importlib
        import config.settings as settings
        importlib.reload(settings)
        import data.storage as storage
        importlib.reload(storage)

        storage.init_db()
        assert (tmp_path / "test.db").exists()

    def test_save_and_retrieve_news(self, tmp_path):
        import os
        os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test2.db"

        import importlib
        import config.settings as settings
        importlib.reload(settings)
        import data.storage as storage
        importlib.reload(storage)

        storage.init_db()

        item = {
            "source": "test",
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "data_type": "news",
            "payload": {
                "uid": "abc123",
                "title": "Test haberi",
                "url": "https://example.com",
                "summary": "Özet",
            },
        }
        storage.save_collector_output(item)
        news = storage.get_recent_news(limit=10)
        assert len(news) == 1
        assert news[0]["title"] == "Test haberi"
