"""Teknik indikatörleri DB'deki BTC verisiyle test eder."""

from data.storage import init_db, get_ohlcv
from data.indicators import prepare_df, compute_all, generate_summary

init_db()

# DB'den BTC 1h verisi çek
candles = get_ohlcv("BTCUSDT", "1h", 200)
print(f"DB'den çekilen mum sayısı: {len(candles)}")

if not candles:
    print("HATA: DB boş. Önce 'py main.py --mode collect-crypto' çalıştırın.")
    exit(1)

# DataFrame'e çevir
df = prepare_df(candles)
print(f"DataFrame: {df.shape[0]} satır, kolonlar: {list(df.columns)}")

# Tüm indikatörleri hesapla
df_ind = compute_all(df)
print(f"\nİndikatör sayısı: {len(df_ind.columns) - 5} (OHLCV hariç)")

# Son satırı göster
last = df_ind.iloc[-1]
print(f"\n--- Son Mum ---")
print(f"Zaman : {df_ind.index[-1]}")
print(f"Kapanış: {last['close']:.2f}")
print(f"RSI    : {last['rsi']:.2f}")
print(f"MACD   : {last['macd']:.4f} | Signal: {last['macd_signal']:.4f}")
print(f"EMA21  : {last['ema_21']:.2f}")
print(f"BB üst : {last['bb_upper']:.2f} | alt: {last['bb_lower']:.2f}")
print(f"ATR    : {last['atr']:.2f}")
print(f"ADX    : {last['adx']:.2f}")

# Doğal dil özeti
print(f"\n{'='*60}")
print("LLM için Doğal Dil Özeti:")
print("="*60)
summary = generate_summary(df_ind, "BTCUSDT")
print(summary)
