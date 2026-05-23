# Kaggle Notebook — 2x T4 DDP ile QLoRA Fine-tune
# Bu notebook 2 hücreye ayrılmıştır:
# Hücre 1: Paket kur + script yaz
# Hücre 2: accelerate launch ile 2x GPU başlat

# ════════════════════════════════════════════════════════════
# HÜCRE 1
# ════════════════════════════════════════════════════════════

import subprocess, os

# Paket kur
subprocess.run([
    "pip", "install", "-q",
    "transformers==4.46.3", "trl==0.12.2", "peft==0.14.0",
    "accelerate", "bitsandbytes", "datasets", "safetensors",
], check=True)

# DDP train script'i diske yaz
SCRIPT = r"""
import os, json, shutil
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from pathlib import Path
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

accelerator = Accelerator()
rank = accelerator.local_process_index   # 0 veya 1

DATASET_DIR = Path("/kaggle/input/datasets/wawdsd/trading-bot-finetune")
OUTPUT_DIR  = Path("/kaggle/working/qwen2.5-1.5b-trading")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
EPOCHS      = 1
BATCH_SIZE  = 8     # her GPU icin -> 2 GPU x 8 = efektif 16 x grad_accum
GRAD_ACCUM  = 2     # efektif batch = 32
MAX_SEQ_LEN = 512

if rank == 0:
    print(f"GPU sayisi: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def load_jsonl(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def sharegpt_to_text(example, tokenizer):
    messages = []
    for c in example["conversations"]:
        role = {"system": "system", "human": "user", "gpt": "assistant"}.get(c["from"], "user")
        messages.append({"role": role, "content": c["value"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)

if rank == 0:
    print("[1/5] Model yukleniyor...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map={"": rank},   # GPU 0 -> rank 0, GPU 1 -> rank 1
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if rank == 0:
    print("[2/5] LoRA ekleniyor...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
))
model.is_loaded_in_4bit = True
model.is_parallelizable = True
model.model_parallel = True

if rank == 0:
    model.print_trainable_parameters()
    print("[3/5] Dataset hazirlaniyor...")

train_examples = load_jsonl(DATASET_DIR / "large_train.jsonl")
val_examples   = load_jsonl(DATASET_DIR / "large_val.jsonl")
train_texts = [sharegpt_to_text(ex, tokenizer) for ex in train_examples]
val_texts   = [sharegpt_to_text(ex, tokenizer) for ex in val_examples]
train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})

if rank == 0:
    print(f"    Train: {len(train_texts)} | Val: {len(val_texts)}")
    print("[4/5] Egitim basliyor...")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=20,
    learning_rate=2e-4,
    fp16=True, bf16=False,
    logging_steps=10,
    save_steps=300,
    eval_steps=300,
    eval_strategy="steps",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    report_to="none",
    dataloader_num_workers=0,
    save_safetensors=True,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
    packing=False,
    args=training_args,
)

trainer_stats = trainer.train()

if rank == 0:
    print("[5/5] Model kaydediliyor...")
    lora_out = OUTPUT_DIR / "lora_weights"
    model.save_pretrained(str(lora_out), safe_serialization=True)
    tokenizer.save_pretrained(str(lora_out))
    runtime = trainer_stats.metrics.get("train_runtime", 0)
    print(f"Sure : {runtime:.0f}s ({runtime/3600:.1f} saat)")
    print(f"Loss : {trainer_stats.metrics.get('train_loss', 0):.4f}")
    print(f"Konum: {lora_out}")
"""

with open("/kaggle/working/train_ddp.py", "w") as f:
    f.write(SCRIPT)

print("Script yazildi: /kaggle/working/train_ddp.py")

# ════════════════════════════════════════════════════════════
# HÜCRE 2  (ayri hücreye yapistir)
# ════════════════════════════════════════════════════════════
# import subprocess
# result = subprocess.run([
#     "accelerate", "launch",
#     "--num_processes", "2",
#     "--mixed_precision", "fp16",
#     "/kaggle/working/train_ddp.py"
# ], check=False)
