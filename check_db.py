from data.storage import get_ohlcv, get_recent_news

candles = get_ohlcv('BTCUSDT', '1h', 5)
print('BTC son 5 mum:')
for c in candles:
    print(f'  {c["open_time"][:16]}  close={c["close"]}')

news = get_recent_news(limit=5)
print(f'\nSon 5 haber:')
for n in news:
    print(f'  [{n["source"]}] {n["title"][:70]}')
