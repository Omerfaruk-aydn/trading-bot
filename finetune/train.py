"""Qwen 2.5-1.5B-Instruct — QLoRA fine-tune (PEFT + TRL, unsloth gerektirmez).

Çalıştırma:
    py -3.11 finetune/train.py --agent combined --epochs 3
"""

import argparse
import json
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets_large"
OUTPUTS_DIR  = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,   # efektif batch = 32
    "warmup_steps": 20,
    "learning_rate": 2e-4,
    "fp16": True,
    "bf16": False,
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 100,
    "max_seq_length": 512,
    "optim": "adamw_8bit",
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
}


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def sharegpt_to_text(example: dict, tokenizer) -> str:
    convs = example["conversations"]
    messages = []
    for c in convs:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(c["from"], "user")
        messages.append({"role": role, "content": c["value"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def train(agent: str, epochs: int, output_name: str | None = None) -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    train_path = DATASETS_DIR / ("large_train.jsonl" if agent == "combined" else f"{agent}_train.jsonl")
    val_path   = DATASETS_DIR / ("large_val.jsonl"   if agent == "combined" else f"{agent}_val.jsonl")

    if not train_path.exists():
        print(f"HATA: {train_path} bulunamadı.")
        return

    print(f"\n{'='*60}")
    print(f"Model : {MODEL_NAME}")
    print(f"Agent : {agent} | Epochs: {epochs}")
    print(f"Train : {train_path}")
    print(f"CUDA  : {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("="*60)

    # 4-bit QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n[1/5] Model yükleniyor (4-bit QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] LoRA adaptörleri ekleniyor...")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=LORA_CONFIG["task_type"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[3/5] Dataset hazırlanıyor...")
    train_examples = load_jsonl(train_path)
    val_examples   = load_jsonl(val_path) if val_path.exists() else []

    train_texts = [sharegpt_to_text(ex, tokenizer) for ex in train_examples]
    val_texts   = [sharegpt_to_text(ex, tokenizer) for ex in val_examples]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset   = Dataset.from_dict({"text": val_texts}) if val_texts else None
    print(f"    Train: {len(train_texts)} | Val: {len(val_texts)}")

    out_dir = OUTPUTS_DIR / (output_name or f"qwen2.5-1.5b-{agent}")

    print("[4/5] Eğitim başlıyor...")
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        fp16=TRAINING_CONFIG["fp16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"] if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=2,
        load_best_model_at_end=bool(val_dataset),
        optim=TRAINING_CONFIG["optim"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        args=training_args,
    )

    trainer_stats = trainer.train()

    print("[5/5] Model kaydediliyor...")
    model.save_pretrained(str(out_dir / "lora_weights"))
    tokenizer.save_pretrained(str(out_dir / "lora_weights"))

    print(f"\nEğitim tamamlandı!")
    print(f"Süre : {trainer_stats.metrics.get('train_runtime', 0):.0f}s "
          f"({trainer_stats.metrics.get('train_runtime', 0)/3600:.1f} saat)")
    print(f"Loss : {trainer_stats.metrics.get('train_loss', 0):.4f}")
    print(f"Model: {out_dir / 'lora_weights'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B QLoRA Fine-tune")
    parser.add_argument("--agent", default="combined",
                        choices=["technical", "news", "risk", "decision", "combined"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    train(args.agent, args.epochs, args.output)
