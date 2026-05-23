"""Dataset pipeline'ını çalıştırır."""
from finetune.dataset_builder import build_all_datasets
from finetune.dataset_stats import print_report

print("Dataset oluşturuluyor...")
counts = build_all_datasets(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    intervals=["1h", "4h", "1d"],
)
print("\nÜretilen örnek sayıları:")
for name, count in counts.items():
    print(f"  {name}: {count}")

print_report()
