"""Modelle terminal sohbeti — fine-tuned Qwen 2.5-1.5B ile piyasa analizi.

Kullanım:
    py agents/chat.py --lora lora_weights/
    py agents/chat.py --lora lora_weights/ --symbol THYAO.IS
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

SYSTEM_PROMPT = (
    "Sen uzman bir Türk borsası (BIST) ve kripto para trading asistanısın. "
    "Teknik analiz, temel analiz ve piyasa haberleri konusunda derin bilgin var. "
    "Türkçe konuşuyorsun. Kullanıcının sorularını net, kısa ve pratik şekilde yanıtlıyorsun. "
    "Gerektiğinde somut al/sat önerisi veriyorsun ama risk uyarısını da ekleyip her zaman ekleyip ekliyorsun."
)


def _load_model(lora_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger.info("Model yükleniyor...")

    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    logger.info("Model hazır. Sohbete başlayabilirsin.\n")
    return model, tokenizer


def _get_market_context(symbol: str | None) -> str:
    if not symbol:
        return ""
    try:
        import yfinance as yf
        from data.indicators import compute_all

        df = yf.download(symbol, period="30d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return ""
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = compute_all(df)
        row = df.iloc[-1]

        return (
            f"\n[{symbol} güncel veri — {df.index[-1].strftime('%d.%m.%Y')}]\n"
            f"Fiyat: {row.get('close', 0):.2f} TL | "
            f"RSI: {row.get('rsi', 0):.1f} | "
            f"MACD: {row.get('macd', 0):.4f} | "
            f"SMA20: {row.get('sma20', 0):.2f} TL\n"
        )
    except Exception:
        return ""


def _generate(model, tokenizer, history: list[dict], user_msg: str) -> str:
    import torch

    history.append({"role": "user", "content": user_msg})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": response})
    return response


def run_chat(lora_path: str, symbol: str | None = None) -> None:
    model, tokenizer = _load_model(lora_path)

    market_ctx = _get_market_context(symbol)
    history: list[dict] = []

    print("=" * 60)
    print("  Trading AI Sohbet — Çıkmak için 'exit' yaz")
    if symbol:
        print(f"  Aktif sembol: {symbol}")
        print(market_ctx)
    print("=" * 60)

    while True:
        try:
            user_input = input("\nSen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "çık", "q"}:
            print("Görüşürüz!")
            break

        # Sembol değiştirme komutu
        if user_input.lower().startswith("/sembol "):
            symbol = user_input.split(" ", 1)[1].strip().upper()
            if not symbol.endswith(".IS") and not symbol.endswith("USDT"):
                symbol += ".IS"
            market_ctx = _get_market_context(symbol)
            print(f"Sembol değişti: {symbol}")
            if market_ctx:
                print(market_ctx)
            continue

        # Güncel piyasa verisini soruya ekle
        full_msg = f"{market_ctx}{user_input}" if market_ctx and len(history) == 0 else user_input

        print("\nAI: ", end="", flush=True)
        response = _generate(model, tokenizer, history, full_msg)
        print(response)

        # Geçmişi max 10 tura sınırla (context taşmasın)
        if len(history) > 20:
            history = history[-20:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading AI Sohbet")
    parser.add_argument("--lora", required=True, help="LoRA adapter klasörü")
    parser.add_argument("--symbol", default=None, help="Başlangıç sembolü (örn: THYAO.IS)")
    args = parser.parse_args()

    run_chat(args.lora, args.symbol)


if __name__ == "__main__":
    main()
