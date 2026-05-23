"""Dataset kalite kontrolü ve istatistik raporu."""

import json
from collections import Counter
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"


def analyze_jsonl(path: Path) -> dict:
    if not path.exists():
        return {"error": "dosya yok"}

    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if not examples:
        return {"count": 0}

    signals, actions, decisions = [], [], []
    token_lens = []

    for ex in examples:
        convs = ex.get("conversations", [])
        gpt_turn = next((c for c in convs if c["from"] == "gpt"), None)
        if not gpt_turn:
            continue
        try:
            out = json.loads(gpt_turn["value"])
        except Exception:
            continue

        # Sinyal dağılımı
        if "signal" in out:
            signals.append(out["signal"])
        if "action" in out:
            actions.append(out["action"])
        if "decision" in out:
            decisions.append(out["decision"])

        # Yaklaşık token sayısı (karakter / 3.5)
        total_text = " ".join(c["value"] for c in convs)
        token_lens.append(int(len(total_text) / 3.5))

    stats = {
        "count": len(examples),
        "avg_tokens": int(sum(token_lens) / len(token_lens)) if token_lens else 0,
        "max_tokens": max(token_lens) if token_lens else 0,
        "min_tokens": min(token_lens) if token_lens else 0,
    }
    if signals:
        stats["signal_dist"] = dict(Counter(signals))
    if actions:
        stats["action_dist"] = dict(Counter(actions))
    if decisions:
        stats["decision_dist"] = dict(Counter(decisions))

    return stats


def print_report():
    print("\n" + "=" * 60)
    print("DATASET KALİTE RAPORU")
    print("=" * 60)

    total = 0
    for name in ["technical", "news", "risk", "decision", "combined"]:
        train_path = DATASETS_DIR / f"{name}_train.jsonl"
        val_path = DATASETS_DIR / f"{name}_val.jsonl"

        train_stats = analyze_jsonl(train_path)
        val_stats = analyze_jsonl(val_path)

        if "error" in train_stats:
            continue

        t_count = train_stats.get("count", 0)
        v_count = val_stats.get("count", 0)
        total += t_count + v_count

        print(f"\n[{name.upper()}]")
        print(f"  Train: {t_count} örnek | Val: {v_count} örnek")
        print(f"  Ort. token: ~{train_stats.get('avg_tokens', 0)} | "
              f"Min: {train_stats.get('min_tokens', 0)} | "
              f"Max: {train_stats.get('max_tokens', 0)}")

        for key in ("signal_dist", "action_dist", "decision_dist"):
            if key in train_stats:
                print(f"  {key}: {train_stats[key]}")

    print(f"\nTOPLAM: {total} örnek")
    print("=" * 60)

    # Örnek görüntüle
    train_path = DATASETS_DIR / "technical_train.jsonl"
    if train_path.exists():
        print("\n--- Teknik Ajan Örnek ---")
        with open(train_path, encoding="utf-8") as f:
            ex = json.loads(f.readline())
        convs = ex["conversations"]
        print(f"[human]: {convs[1]['value'][:200]}...")
        print(f"[gpt]:   {convs[2]['value'][:200]}...")


if __name__ == "__main__":
    print_report()
