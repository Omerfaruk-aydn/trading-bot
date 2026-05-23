"""
Qwen 2.5-1.5B — Haber Sentiment Fine-tuning

Qwen'i finansal haber sentiment sınıflandırması için eğitir.
Çıktı: lora_weights_sentiment/ klasörüne kaydedilir.

Çalıştır:
    py -3.11 finetune/sentiment_train.py
    py -3.11 finetune/sentiment_train.py --epochs 5 --batch 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS_DIR  = Path("finetune/datasets")
OUTPUT_DIR    = Path("lora_weights_sentiment")
MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_PATH    = DATASETS_DIR / "sentiment_train.jsonl"
VAL_PATH      = DATASETS_DIR / "sentiment_val.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def to_text(example: dict, tokenizer) -> str:
    convs = example["conversations"]
    messages = []
    for c in convs:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(c["from"], "user")
        messages.append({"role": role, "content": c["value"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def train(epochs: int = 3, batch_size: int = 8, lr: float = 2e-4) -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    if not TRAIN_PATH.exists():
        print(f"HATA: {TRAIN_PATH} bulunamadi.")
        print("Once veri topla: py -3.11 finetune/news_sentiment_builder.py")
        return

    train_data = load_jsonl(TRAIN_PATH)
    val_data   = load_jsonl(VAL_PATH) if VAL_PATH.exists() else train_data[:50]
    print(f"Train: {len(train_data)} ornek | Val: {len(val_data)} ornek")

    # ── Model yükle ───────────────────────────────────────────────────────────
    print("Model yukleniyor...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA — hafif, sadece sentiment için
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    def _preprocess(examples):
        texts = [to_text(ex, tokenizer) for ex in examples["example"]]
        return tokenizer(texts, truncation=True, max_length=256, padding=False)

    train_ds = Dataset.from_list([{"example": ex} for ex in train_data])
    val_ds   = Dataset.from_list([{"example": ex} for ex in val_data])

    # ── Training ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 32 // batch_size),
        warmup_steps=10,
        learning_rate=lr,
        fp16=True,
        logging_steps=20,
        save_steps=200,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
    )

    def formatting_fn(examples):
        return [to_text(ex, tokenizer) for ex in examples["example"]]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        formatting_func=formatting_fn,
        max_seq_length=256,
    )

    print(f"Egitim baslıyor: {epochs} epoch, batch={batch_size}, lr={lr}")
    trainer.train()

    # ── Kaydet ───────────────────────────────────────────────────────────────
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model kaydedildi: {OUTPUT_DIR}/")
    print("Kullanim: sentiment.py otomatik yükler.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Haber Sentiment Fine-tuning")
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--lr",        type=float, default=2e-4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)


if __name__ == "__main__":
    main()
