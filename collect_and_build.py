"""Master script — veri topla ve büyük dataset oluştur.

Kullanım:
    py collect_and_build.py --step collect   # Tüm piyasalardan veri çek
    py collect_and_build.py --step dataset   # Dataset oluştur
    py collect_and_build.py --step all       # İkisini birden yap

Veri toplama saatlerce sürebilir. Ctrl+C ile durdurup
aynı komutla devam edebilirsiniz (checkpoint mevcuttur).
"""

import argparse
from loguru import logger

from data.storage import init_db


def run_collect(args) -> None:
    from data.collectors.bulk_collector import run_full_collection
    import os

    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        logger.warning("FRED_API_KEY tanımlı değil — makro veri atlanacak.")
        logger.warning("Ücretsiz key: https://fred.stlouisfed.org/docs/api/api_key.html")

    run_full_collection(
        include_crypto=not args.skip_crypto,
        include_stocks=not args.skip_stocks,
        include_macro=not args.skip_macro and bool(fred_key),
        fred_api_key=fred_key or None,
    )


def run_dataset(args) -> None:
    from finetune.large_dataset_builder import build_large_dataset

    logger.info("Dataset oluşturuluyor — hedef: {:,} örnek", args.target)
    counts = build_large_dataset(
        target_examples=args.target,
        step=args.step_size,
    )
    logger.info("\n=== DATASET RAPORU ===")
    for k, v in counts.items():
        logger.info("  {}: {:,}", k, v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Bot — Veri & Dataset Pipeline")
    parser.add_argument("--step", choices=["collect", "dataset", "all"], default="all")
    parser.add_argument("--skip-crypto", action="store_true")
    parser.add_argument("--skip-stocks", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--target", type=int, default=150_000, help="Hedef örnek sayısı")
    parser.add_argument("--step-size", type=int, default=3, dest="step_size",
                        help="Kaç mumda bir örnek (küçük = daha fazla örnek)")
    args = parser.parse_args()

    init_db()

    if args.step in ("collect", "all"):
        logger.info("AŞAMA 1: Veri toplama başlıyor...")
        run_collect(args)

    if args.step in ("dataset", "all"):
        logger.info("AŞAMA 2: Dataset oluşturuluyor...")
        run_dataset(args)


if __name__ == "__main__":
    main()
