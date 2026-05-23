"""Trading Bot CLI — interaktif komut arayüzü."""

from __future__ import annotations

import sys
import os

# Windows terminali UTF-8 yapılandır (kutu çizimi karakterleri için)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         TRADING BOT — AI Destekli Multi-Market Trader    ║
║    BIST · Kripto · Futures · VİOP · NYSE/NASDAQ · LLM    ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
[bold cyan]Komutlar:[/bold cyan]

  [green]start[/green]      — Trading agent başlat
  [green]chat[/green]       — AI ile sohbet et
  [green]backtest[/green]   — Geçmiş veri testi
  [green]status[/green]     — Portföy durumu
  [green]movers[/green]     — Piyasa hareketlileri (kazananlar/kaybedenler/en aktifler)
  [green]train-ml[/green]        — ML sinyal modelini eğit (XGBoost)
  [green]ml-predict[/green]     — Belirli hisseler için ML sinyali göster
  [green]train-sentiment[/green] — Haber sentiment modeli eğit (Qwen fine-tune)
  [green]analyze[/green]         — Hisse/kripto derin analizi (2 yıllık grafik + AI yorum)
  [green]scan[/green]            — Piyasa tara, en güçlü AL/SAT sinyallerini listele
  [green]help[/green]       — Bu yardım
  [green]exit[/green]       — Çık

[bold cyan]start seçenekleri:[/bold cyan]
  --capital   100000          Başlangıç sermayesi (TL)
  --target    20              Hedef getiri %
  --mode      normal          Risk modu: conservative / normal / aggressive / scalping
  --symbols   THYAO GARAN     BIST hisseleri (varsayılan: 8 büyük hisse)
  --crypto                    Kripto spot ekle (BTC-USD ETH-USD SOL-USD)
  --futures                   Binance Futures ekle (BTC-PERP ETH-PERP SOL-PERP)
  --viop                      VİOP vadeli sözleşmeler (XU030-FUT USDTRY-FUT)
  --us                        NYSE/NASDAQ hisseleri (AAPL MSFT NVDA ...)
  --leverage  20              Futures/VİOP kaldıraç (varsayılan: 20)
  --universe                  Tüm borsaları dinamik tara (S&P500, NASDAQ, tüm kripto, BIST 100)
  --top-n     20              Evren modunda kategori başına kaç sembol

[bold cyan]Piyasa kombinasyonları:[/bold cyan]
  [dim]start --crypto                      # Sadece kripto spot[/dim]
  [dim]start --futures --leverage 10       # Sadece kripto futures, 10x[/dim]
  [dim]start --viop                        # Sadece VİOP vadeli[/dim]
  [dim]start --us                          # Sadece ABD hisseleri[/dim]
  [dim]start --symbols THYAO GARAN --us    # BIST + ABD hisseleri[/dim]
  [dim]start --crypto --futures            # Kripto spot + futures[/dim]
  [dim]start --universe                   # TÜM borsalar — S&P500 + NASDAQ + kripto + BIST[/dim]
  [dim]start --universe --top-n 30        # Evren modu, kategori başına 30 hisse[/dim]

[bold cyan]backtest seçenekleri:[/bold cyan]
  --symbols   THYAO GARAN ...
  --period    3mo        1mo / 3mo / 6mo / 1y / 2y
  --mode      normal

[bold cyan]chat seçenekleri:[/bold cyan]
  --symbol    THYAO.IS   Başlangıç sembolü (isteğe bağlı)

[bold cyan]train-ml seçenekleri:[/bold cyan]
  --market    us         Hangi piyasa: us / bist / crypto / all (varsayılan: us)
  --period    2y         Kaç yıllık veri: 1y / 2y / 5y / max (varsayılan: 2y)
  --horizon   5          Kaç gün ilerisi hedef (varsayılan: 5)
  --buy-thr   0.03       AL eşiği — %3 artış = 0.03 (varsayılan: 0.03)
  --max-sym   200        Hızlı test için sembol sayısı sınırla

[bold cyan]analyze seçenekleri:[/bold cyan]
  --symbol    AKBNK          Hisse/kripto sembolü (BIST için .IS opsiyonel)
  --period    2y             Kaç yıllık veri: 1y / 2y (varsayılan: 2y)
  --no-qwen                  Qwen yorumu olmadan sadece teknik raporu göster

[bold cyan]scan seçenekleri:[/bold cyan]
  --market    bist           Hangi piyasa: bist / us / crypto / all (varsayılan: bist)
  --signal    buy            Filtre: buy / sell / all (varsayılan: buy)
  --top       15             Kaç sonuç göster (varsayılan: 15)

[bold cyan]ml-predict seçenekleri:[/bold cyan]
  --symbols   AAPL MSFT  Tahmin yapılacak semboller
"""

DEFAULT_LORA    = "lora_weights"
DEFAULT_SYMBOLS = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS",
                   "AKBNK.IS", "YKBNK.IS", "KCHOL.IS", "SISE.IS"]
DEFAULT_CRYPTO   = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_FUTURES  = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
DEFAULT_VIOP     = ["XU030-FUT", "USDTRY-FUT"]
DEFAULT_US       = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
                    "META", "TSLA", "AMD", "NFLX", "INTC"]

_agent = None
_agent_thread = None


def _check_lora() -> bool:
    lora = Path(DEFAULT_LORA)
    if not lora.exists() or not (lora / "adapter_model.safetensors").exists():
        console.print(
            Panel(
                f"[red]LoRA adapter bulunamadı![/red]\n"
                f"Beklenen konum: [cyan]{lora.absolute()}[/cyan]\n\n"
                "Kaggle'dan indirdiğin dosyaları bu klasöre koy:\n"
                "  adapter_model.safetensors, adapter_config.json, tokenizer.json vb.",
                title="Hata", border_style="red",
            )
        )
        return False
    return True


def _parse_inline(line: str) -> argparse.Namespace:
    """Komut satırını parse et: 'start --capital 50000 --mode aggressive'"""
    parts = line.strip().split()
    cmd = parts[0] if parts else ""
    rest = parts[1:] if len(parts) > 1 else []

    p = argparse.ArgumentParser(exit_on_error=False)
    if cmd == "start":
        p.add_argument("--capital",  type=float, default=100_000)
        p.add_argument("--target",   type=float, default=20.0)
        p.add_argument("--mode",     default="normal")
        p.add_argument("--symbols",  nargs="*", default=None)
        p.add_argument("--crypto",   action="store_true")
        p.add_argument("--futures",  action="store_true")
        p.add_argument("--viop",     action="store_true")
        p.add_argument("--us",       action="store_true")
        p.add_argument("--leverage", type=int, default=20)
        p.add_argument("--crypto-symbols",  nargs="*", default=None)
        p.add_argument("--futures-symbols", nargs="*", default=None)
        p.add_argument("--viop-symbols",    nargs="*", default=None)
        p.add_argument("--us-symbols",      nargs="*", default=None)
        p.add_argument("--interval", type=int, default=None)
        p.add_argument("--universe", action="store_true",
                       help="Tüm borsaları dinamik tarama — en çok hareket eden hisseleri otomatik seç")
        p.add_argument("--top-n", type=int, default=20,
                       help="Evren modunda kategori başına kaç sembol (varsayılan: 20)")
    elif cmd == "backtest":
        p.add_argument("--symbols", nargs="*", default=["THYAO.IS", "GARAN.IS", "ASELS.IS"])
        p.add_argument("--capital", type=float, default=100_000)
        p.add_argument("--period", default="3mo")
        p.add_argument("--mode", default="normal")
    elif cmd == "chat":
        p.add_argument("--symbol", default=None)
    elif cmd == "train-ml":
        p.add_argument("--market",  default="us",
                       choices=["us", "bist", "crypto", "all"])
        p.add_argument("--period",  default="2y")
        p.add_argument("--horizon", type=int,   default=5)
        p.add_argument("--buy-thr", type=float, default=0.03, dest="buy_thr")
        p.add_argument("--max-sym", type=int,   default=None, dest="max_sym")
    elif cmd == "ml-predict":
        p.add_argument("--symbols", nargs="*",  default=["AAPL", "MSFT", "NVDA"])
        p.add_argument("--period",  default="1y")
        p.add_argument("--threshold", type=float, default=0.55)
    elif cmd == "analyze":
        p.add_argument("--symbol",   required=True, help="Hisse/kripto sembolü (ör: AKBNK veya AKBNK.IS)")
        p.add_argument("--period",   default="2y",  choices=["1y", "2y"])
        p.add_argument("--no-qwen",  action="store_true", dest="no_qwen")
    elif cmd == "scan":
        p.add_argument("--market",  default="bist", choices=["bist", "us", "crypto", "all"])
        p.add_argument("--signal",  default="buy",  choices=["buy", "sell", "all"])
        p.add_argument("--top",     type=int, default=15, dest="top_n")
    elif cmd == "train-sentiment":
        p.add_argument("--symbols",  default="all", choices=["all", "bist", "us", "crypto"])
        p.add_argument("--epochs",   type=int,   default=3)
        p.add_argument("--batch",    type=int,   default=8)
        p.add_argument("--skip-data", action="store_true", dest="skip_data",
                       help="Veri toplamayı atla (dataset zaten varsa)")

    try:
        args = p.parse_args(rest)
        args.cmd = cmd
        return args
    except Exception:
        args = argparse.Namespace(cmd=cmd)
        return args


def cmd_start(args: argparse.Namespace) -> None:
    global _agent, _agent_thread

    if _agent and getattr(_agent, "_running", False):
        console.print("[yellow]Agent zaten çalışıyor. Önce durdurun (Ctrl+C).[/yellow]")
        return

    if not _check_lora():
        return

    # ── Piyasa tespiti ────────────────────────────────────────────────────────
    any_market = any([
        args.crypto, getattr(args, "futures", False),
        getattr(args, "viop", False), getattr(args, "us", False),
    ])

    # Eğer sadece özel piyasalar seçildiyse ve --symbols verilmediyse BIST sıfır
    if any_market and not args.symbols:
        symbols = []
    else:
        symbols = [s if s.endswith(".IS") else f"{s}.IS"
                   for s in (args.symbols or DEFAULT_SYMBOLS)]

    crypto   = (getattr(args, "crypto_symbols",  None) or DEFAULT_CRYPTO)  if args.crypto              else None
    futures  = (getattr(args, "futures_symbols", None) or DEFAULT_FUTURES) if getattr(args, "futures", False) else None
    viop     = (getattr(args, "viop_symbols",    None) or DEFAULT_VIOP)    if getattr(args, "viop",    False) else None
    us       = (getattr(args, "us_symbols",      None) or DEFAULT_US)      if getattr(args, "us",      False) else None
    leverage = getattr(args, "leverage", 20)

    from agents.llm_trading_agent import LLMTradingAgent

    universe_mode = getattr(args, "universe", False)
    top_n = getattr(args, "top_n", 20)

    _agent = LLMTradingAgent(
        lora_path=DEFAULT_LORA,
        symbols=symbols,
        initial_capital=args.capital,
        target_pct=args.target,
        mode=args.mode,
        scan_interval=args.interval,
        crypto_symbols=crypto,
        futures_symbols=futures,
        viop_symbols=viop,
        us_symbols=us,
        leverage=leverage,
        universe_mode=universe_mode,
        universe_top_n=top_n,
    )

    def _run():
        _agent.run()

    _agent_thread = threading.Thread(target=_run, daemon=True)
    _agent_thread.start()

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_row("[cyan]Sermaye[/cyan]",   f"{args.capital:,.0f} TL")
    table.add_row("[cyan]Hedef[/cyan]",     f"%{args.target:.0f} ({args.capital*(1+args.target/100):,.0f} TL)")
    table.add_row("[cyan]Mod[/cyan]",       args.mode.upper())
    if universe_mode:
        table.add_row("[cyan]Evren Modu[/cyan]",
                      f"[bold green]AÇIK[/bold green] — tüm borsalar, kategori başına TOP {top_n}")
    if symbols:
        table.add_row("[cyan]BIST[/cyan]",  ", ".join(symbols))
    if crypto:
        table.add_row("[cyan]Kripto[/cyan]",    ", ".join(crypto))
    if futures:
        table.add_row("[cyan]Futures[/cyan]",   f"{', '.join(futures)} ({leverage}x kaldıraç)")
    if viop:
        table.add_row("[cyan]VİOP[/cyan]",      f"{', '.join(viop)} ({leverage}x kaldıraç)")
    if us:
        table.add_row("[cyan]NYSE/NASDAQ[/cyan]", ", ".join(us))

    console.print(Panel(table, title="[green]Agent Başlatıldı[/green]", border_style="green"))
    console.print("[dim]Agent arka planda çalışıyor. 'status' ile takip edebilirsin.[/dim]")


def cmd_chat(args: argparse.Namespace) -> None:
    if not _check_lora():
        return

    from agents.chat import run_chat
    symbol = getattr(args, "symbol", None)
    console.print(Panel(
        "Modelle sohbet başlıyor...\n[dim]Çıkmak için 'exit' yaz[/dim]",
        border_style="cyan",
    ))
    run_chat(DEFAULT_LORA, symbol)


def cmd_backtest(args: argparse.Namespace) -> None:
    symbols = [s if s.endswith(".IS") else f"{s}.IS" for s in args.symbols]
    console.print(Panel(
        f"Backtest başlatılıyor | {', '.join(symbols)} | {args.period} | {args.mode.upper()}",
        border_style="yellow",
    ))
    from agents.backtest import run_backtest
    run_backtest(symbols, args.capital, args.period, args.mode)


def cmd_movers() -> None:
    """Tüm piyasalardaki en çok hareket eden hisseleri gösterir."""
    from data.sources.market_screener import screen_us, screen_crypto, screen_bist
    from data.markets.us_stocks import session_label, get_us_session

    sess = get_us_session()
    sess_color = {"regular": "green", "premarket": "yellow", "aftermarket": "cyan", "closed": "red"}[sess]
    console.print(f"[bold {sess_color}]ABD Seans: {session_label()}[/bold {sess_color}]")
    if sess in ("premarket", "aftermarket"):
        console.print("[yellow]Extended hours — TÜM NYSE+NASDAQ (~5,000 hisse) taranıyor, biraz bekle...[/yellow]")
    else:
        console.print("[dim]Piyasalar taranıyor...[/dim]")

    sections = [
        ("ABD Hisseleri", screen_us, "us"),
        ("Kripto (Binance)", screen_crypto, "crypto"),
        ("BIST", screen_bist, "bist"),
    ]

    for title, fn, _mtype in sections:
        try:
            data = fn(top_n=15)
        except Exception as e:
            console.print(f"[red]{title} taranamadı: {e}[/red]")
            continue

        for category, label, color in [
            ("gainers",     "KAZANANLAR", "green"),
            ("losers",      "KAYBEDİLER", "red"),
            ("most_active", "EN AKTİF",   "cyan"),
        ]:
            quotes = data.get(category, [])
            if not quotes:
                continue
            t = Table(
                title=f"[bold {color}]{title} — {label}[/bold {color}]",
                box=box.SIMPLE,
                show_header=True,
            )
            t.add_column("Sembol",   style="white",  min_width=12)
            t.add_column("Fiyat",    justify="right", style="white")
            t.add_column("24h %",    justify="right")
            t.add_column("Hacim",    justify="right", style="dim")

            for q in quotes[:10]:
                chg_color = "green" if q.change_pct >= 0 else "red"
                sign      = "+" if q.change_pct >= 0 else ""
                vol_str   = (
                    f"${q.volume_usd/1e9:.1f}B" if q.volume_usd >= 1e9
                    else f"${q.volume_usd/1e6:.1f}M" if q.volume_usd >= 1e6
                    else f"${q.volume_usd/1e3:.0f}K"
                )
                t.add_row(
                    q.symbol,
                    f"{q.price:,.4f}" if q.price < 1 else f"{q.price:,.2f}",
                    f"[{chg_color}]{sign}{q.change_pct:.2f}%[/{chg_color}]",
                    vol_str,
                )
            console.print(t)


def cmd_train_ml(args: argparse.Namespace) -> None:
    """ML sinyal modelini eğitir."""
    from data.sources.stock_universe import (
        get_full_us_universe, get_bist_symbols, get_crypto_universe,
    )
    from ml.data_pipeline import build_dataset
    from ml.trainer import train

    market = getattr(args, "market", "us")
    period  = getattr(args, "period",  "2y")
    horizon = getattr(args, "horizon", 5)
    buy_thr = getattr(args, "buy_thr", 0.03)
    max_sym = getattr(args, "max_sym", None)

    console.print(Panel(
        f"[bold cyan]ML Modeli Eğitimi Başlıyor[/bold cyan]\n"
        f"Piyasa: [yellow]{market.upper()}[/yellow] | "
        f"Dönem: [yellow]{period}[/yellow] | "
        f"Hedef: [yellow]{horizon} gün[/yellow] | "
        f"AL eşiği: [yellow]{buy_thr*100:.0f}%[/yellow]"
        + (f" | Maks sembol: [yellow]{max_sym}[/yellow]" if max_sym else ""),
        border_style="cyan",
    ))

    # Sembol listesi seç
    if market == "us":
        symbols = get_full_us_universe()
        console.print(f"[dim]ABD evreni: {len(symbols)} sembol[/dim]")
    elif market == "bist":
        symbols = get_bist_symbols()
        console.print(f"[dim]BIST: {len(symbols)} sembol[/dim]")
    elif market == "crypto":
        raw = get_crypto_universe()
        # yfinance formatı: BTC-USD
        symbols = raw
        console.print(f"[dim]Kripto: {len(symbols)} çift[/dim]")
    else:  # all
        symbols = get_full_us_universe() + get_bist_symbols()
        console.print(f"[dim]Tüm piyasalar: {len(symbols)} sembol[/dim]")

    console.print("[dim]Veri indiriliyor ve özellikler hesaplanıyor...[/dim]")

    try:
        X, y = build_dataset(
            symbols,
            period=period,
            horizon=horizon,
            buy_thr=buy_thr,
            max_symbols=max_sym,
        )
        console.print(f"[green]Dataset hazır: {len(X):,} örnek[/green]")
        console.print("[dim]Model eğitiliyor...[/dim]")
        model = train(X, y, save=True, market=market)
        console.print(Panel(
            "[bold green]Model eğitimi tamamlandı![/bold green]\n"
            f"Model [cyan]ml/models/signal_model_{market}.pkl[/cyan] dosyasına kaydedildi.\n"
            "Artık bot otomatik olarak bu modeli kullanacak.",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[red]Eğitim hatası: {e}[/red]")
        import traceback
        traceback.print_exc()


def cmd_ml_predict(args: argparse.Namespace) -> None:
    """Belirli semboller için ML sinyal tahmini gösterir."""
    import warnings, logging
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    from ml.predictor import get_predictor

    predictor = get_predictor()
    if not predictor.available:
        console.print("[red]Model bulunamadı — önce 'train-ml' çalıştır.[/red]")
        return

    symbols  = getattr(args, "symbols", ["AAPL", "MSFT", "NVDA"])
    period   = getattr(args, "period", "1y")
    thr      = getattr(args, "threshold", 0.55)

    t = Table(
        title="[bold cyan]ML Sinyal Tahminleri[/bold cyan]",
        box=box.ROUNDED, border_style="cyan",
    )
    t.add_column("Sembol",     min_width=10)
    t.add_column("Sinyal",     justify="center")
    t.add_column("Olasılık",   justify="right")
    t.add_column("Yorum",      style="dim")

    for sym in symbols:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, __import__("pandas").MultiIndex):
                df.columns = df.columns.get_level_values(0)
            signal, conf = predictor.predict(df, threshold=thr)
            sig_text  = "[bold green]AL[/bold green]" if signal == 1 else "[dim]BEKLE[/dim]"
            conf_color = "green" if conf >= 0.65 else "yellow" if conf >= 0.55 else "red"
            yorum = (
                "Güçlü AL sinyali" if conf >= 0.70 else
                "AL sinyali" if conf >= 0.55 else
                "Zayıf / belirsiz"
            )
            t.add_row(
                sym, sig_text,
                f"[{conf_color}]{conf:.2%}[/{conf_color}]",
                yorum,
            )
        except Exception as e:
            t.add_row(sym, "[red]HATA[/red]", "-", str(e)[:40])

    console.print(t)
    console.print(f"[dim]Eşik: {thr:.0%} | Dönem: {period}[/dim]")


def cmd_status() -> None:
    global _agent

    if not _agent:
        console.print("[yellow]Agent çalışmıyor.[/yellow]")
        return

    total = _agent._total_value()
    portfolio = _agent.portfolio
    pnl = total - _agent.initial_capital
    pnl_pct = pnl / _agent.initial_capital * 100
    target = _agent.target_value
    progress = min(total / target * 100, 100)

    table = Table(box=box.ROUNDED, border_style="cyan")
    table.add_column("", style="cyan")
    table.add_column("", style="white")

    table.add_row("Toplam Değer",   f"{total:>12,.0f} TL")
    table.add_row("Nakit",          f"{portfolio.cash:>12,.0f} TL")
    table.add_row("P&L",            f"[{'green' if pnl >= 0 else 'red'}]{pnl:>+12,.0f} TL ({pnl_pct:+.2f}%)[/]")
    table.add_row("Hedef",          f"{target:>12,.0f} TL")
    table.add_row("İlerleme",       f"%{progress:.1f}")
    table.add_row("Spot pozisyon",  str(len(portfolio.positions)))
    table.add_row("Futures pozisyon", str(len(_agent.futures_positions)))
    table.add_row("Toplam işlem",   str(len(portfolio.trades)))

    console.print(Panel(table, title="[cyan]Portföy Durumu[/cyan]"))

    # ── Spot pozisyonlar ──────────────────────────────────────────────────────
    if portfolio.positions:
        pos_table = Table(title="Spot Pozisyonlar", box=box.SIMPLE)
        pos_table.add_column("Sembol")
        pos_table.add_column("Tür")
        pos_table.add_column("Adet", justify="right")
        pos_table.add_column("Giriş", justify="right")
        pos_table.add_column("Güncel", justify="right")
        pos_table.add_column("P&L", justify="right")

        for sym, pos in portfolio.positions.items():
            cur_price = _agent._get_price(sym)
            if cur_price <= 0:
                cur_price = pos.entry_price
            p = pos.pnl(cur_price)
            color = "green" if p >= 0 else "red"
            from agents.llm_trading_agent import _market_type, _currency
            mt = _market_type(sym)
            unit = _currency(sym)
            pos_table.add_row(
                sym, mt.upper(),
                f"{pos.shares:.4f}",
                f"{pos.entry_price:.4f} {unit}",
                f"{cur_price:.4f} {unit}",
                f"[{color}]{p:+,.0f} TL[/]",
            )
        console.print(pos_table)

    # ── Futures pozisyonlar ───────────────────────────────────────────────────
    if _agent.futures_positions:
        fut_table = Table(title="Futures / VİOP Pozisyonlar", box=box.SIMPLE)
        fut_table.add_column("Sembol")
        fut_table.add_column("Yön")
        fut_table.add_column("Miktar", justify="right")
        fut_table.add_column("Giriş", justify="right")
        fut_table.add_column("Güncel", justify="right")
        fut_table.add_column("Likidasyon", justify="right")
        fut_table.add_column("P&L", justify="right")

        from agents.llm_trading_agent import _currency
        from data.markets.viop import VIOP_CONTRACTS
        for sym, fpos in _agent.futures_positions.items():
            cur_price = _agent._get_price(sym)
            if cur_price <= 0:
                cur_price = fpos.entry_price
            mult = VIOP_CONTRACTS[sym].multiplier if sym in VIOP_CONTRACTS else 1.0
            pnl_tl = fpos.unrealized_pnl(cur_price, mult)
            color = "green" if pnl_tl >= 0 else "red"
            unit = _currency(sym)
            side_color = "green" if fpos.side == "long" else "red"
            fut_table.add_row(
                sym,
                f"[{side_color}]{fpos.side.upper()}[/] ({fpos.leverage}x)",
                f"{fpos.contracts:.4f}",
                f"{fpos.entry_price:.4f} {unit}",
                f"{cur_price:.4f} {unit}",
                f"[red]{fpos.liquidation_price:.4f}[/]",
                f"[{color}]{pnl_tl:+,.0f} TL[/]",
            )
        console.print(fut_table)


def main() -> None:
    console.print(BANNER, style="bold cyan")
    console.print("[dim]'help' yazarak komutları görebilirsin.[/dim]\n")

    while True:
        try:
            raw = console.input("[bold cyan]trading-bot>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Görüşürüz![/dim]")
            break

        if not raw:
            continue

        cmd = raw.split()[0].lower()

        if cmd in {"exit", "quit", "çık", "q"}:
            console.print("[dim]Görüşürüz![/dim]")
            break

        elif cmd == "help":
            console.print(Panel(HELP_TEXT, title="Yardım", border_style="cyan"))

        elif cmd == "start":
            try:
                args = _parse_inline(raw)
                cmd_start(args)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "chat":
            try:
                args = _parse_inline(raw)
                cmd_chat(args)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "backtest":
            try:
                args = _parse_inline(raw)
                cmd_backtest(args)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "status":
            try:
                cmd_status()
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "movers":
            try:
                cmd_movers()
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "train-ml":
            try:
                args = _parse_inline(raw)
                cmd_train_ml(args)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "ml-predict":
            try:
                args = _parse_inline(raw)
                cmd_ml_predict(args)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "analyze":
            try:
                args = _parse_inline(raw)
                from agents.analyzer import run_analyze
                lora = DEFAULT_LORA if not args.no_qwen and _check_lora() else None
                run_analyze(args.symbol, period=args.period, lora_path=lora)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "scan":
            try:
                args = _parse_inline(raw)
                from agents.scanner import run_scan
                run_scan(market=args.market, signal=args.signal, top_n=args.top_n)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        elif cmd == "train-sentiment":
            try:
                args = _parse_inline(raw)
                from pathlib import Path as _Path
                train_path = _Path("finetune/datasets/sentiment_train.jsonl")
                if not args.skip_data or not train_path.exists():
                    console.print("[cyan]Haber verisi toplaniyor...[/cyan]")
                    from finetune.news_sentiment_builder import build_dataset
                    markets = ["bist", "us", "crypto"] if args.symbols == "all" else [args.symbols]
                    build_dataset(markets=markets, out_path=train_path)
                console.print("[cyan]Qwen sentiment fine-tuning basliyor...[/cyan]")
                from finetune.sentiment_train import train as sentiment_train
                sentiment_train(epochs=args.epochs, batch_size=args.batch)
            except Exception as e:
                console.print(f"[red]Hata: {e}[/red]")

        else:
            console.print(f"[red]Bilinmeyen komut: '{cmd}'[/red] — 'help' yaz")


if __name__ == "__main__":
    main()
