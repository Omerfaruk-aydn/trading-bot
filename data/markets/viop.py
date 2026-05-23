"""VİOP (Borsa İstanbul Vadeli İşlemler Piyasası) sözleşme tanımları ve yardımcı fonksiyonlar."""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger

import yfinance as yf


@dataclass(frozen=True)
class ViopContract:
    symbol: str          # İç sembol: "XU030-FUT"
    name: str            # Açıklama
    underlying_yf: str   # yfinance dayanak varlık sembolü
    multiplier: float    # 1 sözleşme = multiplier × fiyat TL
    tick_size: float     # Minimum fiyat adımı
    margin_pct: float    # Başlangıç teminat oranı (% olarak, ör: 10.0 = %10)
    contract_type: str   # "index" | "stock" | "forex" | "commodity"


VIOP_CONTRACTS: dict[str, ViopContract] = {
    "XU030-FUT": ViopContract(
        symbol="XU030-FUT",
        name="BIST30 Endeks Vadeli",
        underlying_yf="XU030.IS",
        multiplier=10.0,
        tick_size=0.25,
        margin_pct=10.0,
        contract_type="index",
    ),
    "USDTRY-FUT": ViopContract(
        symbol="USDTRY-FUT",
        name="USD/TRY Döviz Vadeli",
        underlying_yf="USDTRY=X",
        multiplier=1000.0,
        tick_size=0.0001,
        margin_pct=8.0,
        contract_type="forex",
    ),
    "EURTRY-FUT": ViopContract(
        symbol="EURTRY-FUT",
        name="EUR/TRY Döviz Vadeli",
        underlying_yf="EURTRY=X",
        multiplier=1000.0,
        tick_size=0.0001,
        margin_pct=8.0,
        contract_type="forex",
    ),
    "GOLD-FUT": ViopContract(
        symbol="GOLD-FUT",
        name="Altın Vadeli (gram TL)",
        underlying_yf="GC=F",
        multiplier=100.0,
        tick_size=0.01,
        margin_pct=12.0,
        contract_type="commodity",
    ),
    "THYAO-FUT": ViopContract(
        symbol="THYAO-FUT",
        name="Türk Hava Yolları Hisse Vadeli",
        underlying_yf="THYAO.IS",
        multiplier=100.0,
        tick_size=0.01,
        margin_pct=15.0,
        contract_type="stock",
    ),
    "GARAN-FUT": ViopContract(
        symbol="GARAN-FUT",
        name="Garanti Bankası Hisse Vadeli",
        underlying_yf="GARAN.IS",
        multiplier=100.0,
        tick_size=0.01,
        margin_pct=15.0,
        contract_type="stock",
    ),
}

DEFAULT_VIOP = ["XU030-FUT", "USDTRY-FUT"]


def is_viop(symbol: str) -> bool:
    return symbol in VIOP_CONTRACTS


def get_viop_price(symbol: str) -> float:
    """Dayanak varlığın güncel fiyatı (mark price yaklaşımı olarak kullanılır)."""
    contract = VIOP_CONTRACTS.get(symbol)
    if not contract:
        return 0.0
    try:
        t = yf.Ticker(contract.underlying_yf)
        price = float(t.fast_info.last_price or 0)
        return price
    except Exception as e:
        logger.debug("VİOP fiyat hatası ({}/{}): {}", symbol, contract.underlying_yf, e)
        return 0.0


def calc_liquidation_price(
    entry: float, side: str, leverage: float, maint_rate: float = 0.004
) -> float:
    """
    Tasfiye (likidasyon) fiyatı hesapla.

    Formül (Binance USDT-M Tier-1 yaklaşımı):
      Long:  entry × (1 - 1/leverage + maint_rate)
      Short: entry × (1 + 1/leverage - maint_rate)

    maint_rate varsayılan 0.004 (%0.4) — Binance BTC Tier-1 bakım teminatı.
    VİOP için sözleşmeye göre değişebilir; konservatif yaklaşım için 0.05 kullanılabilir.
    """
    lev = max(leverage, 1.0)
    if side == "long":
        return entry * (1 - 1 / lev + maint_rate)
    return entry * (1 + 1 / lev - maint_rate)
