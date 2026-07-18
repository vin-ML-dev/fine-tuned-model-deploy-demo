"""
Phase 1 - Step 3/4: Preprocess the generated dataset.

- Cleans whitespace
- Removes exact and near-duplicate questions
- Performs a stratified train / validation / test split (per section)
- Writes train.jsonl, val.jsonl, test.jsonl + a dataset_stats.json report

Usage:
    python scripts/preprocess.py
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "dataset_full.jsonl"
OUT_DIR = ROOT / "data" / "processed"

TRAIN_FRAC, VAL_FRAC = 0.85, 0.10  # test gets the remainder (0.05)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace - used only for dedup keys."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"{IN_PATH} not found. Run scripts/generate_dataset.py first."
        )

    rows = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    print(f"Loaded {len(rows)} raw examples")

    # ---- Clean + dedup on normalized user question ----
    seen = set()
    cleaned = []
    for r in rows:
        for m in r["messages"]:
            m["content"] = clean_text(m["content"])
        user_q = next(m["content"] for m in r["messages"] if m["role"] == "user")
        key = normalize(user_q)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(r)
    print(f"After dedup: {len(cleaned)} examples ({len(rows) - len(cleaned)} removed)")

    # ---- Stratified split by section ----
    by_section = defaultdict(list)
    for r in cleaned:
        by_section[r["section"]].append(r)

    train, val, test = [], [], []
    for section, items in by_section.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * TRAIN_FRAC))
        n_val = max(1, int(n * VAL_FRAC)) if n > 2 else 0
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    splits = {"train": train, "val": val, "test": test}
    for name, data in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} examples -> {path}")

    # ---- Stats report (data versioning aid) ----
    stats = {
        "total_after_dedup": len(cleaned),
        "splits": {k: len(v) for k, v in splits.items()},
        "sections": {k: len(v) for k, v in sorted(by_section.items())},
        "avg_answer_chars": round(
            sum(len(r["messages"][2]["content"]) for r in cleaned) / len(cleaned), 1
        ),
    }
    with open(OUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Stats written to dataset_stats.json")


if __name__ == "__main__":
    main()
