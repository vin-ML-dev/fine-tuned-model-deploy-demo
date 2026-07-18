"""
Phase 2 - Step 9: Evaluate the fine-tuned adapter on the held-out test set.

Two checks:
  1. Test-set loss / perplexity (quantitative)
  2. Side-by-side generations: base model vs fine-tuned model (qualitative)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --num_samples 5
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)


def load_model(base_model, adapter_path=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, messages, max_new_tokens=200):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def test_loss(model, tokenizer, rows, max_len=512):
    """Average causal LM loss over the full chat text of each test example."""
    losses = []
    for r in rows:
        text = tokenizer.apply_chat_template(r["messages"], tokenize=False)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        losses.append(out.loss.item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/training_config.yaml"))
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--compare_base", action="store_true",
                        help="Also generate answers with the raw base model")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base_model = cfg["model"]["base_model"]
    adapter = ROOT / cfg["training"]["output_dir"] / "final_adapter"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_rows = [json.loads(l) for l in open(ROOT / cfg["data"]["test_file"], encoding="utf-8")]
    print(f"Test set size: {len(test_rows)}")

    # ---------- Fine-tuned model ----------
    print("\nLoading fine-tuned model...")
    ft_model = load_model(base_model, str(adapter))
    loss, ppl = test_loss(ft_model, tokenizer, test_rows)
    print(f"[fine-tuned] test loss: {loss:.4f} | perplexity: {ppl:.2f}")

    # ---------- Sample generations ----------
    samples = random.sample(test_rows, min(args.num_samples, len(test_rows)))
    results = []
    for r in samples:
        prompt_msgs = r["messages"][:2]  # system + user
        question = prompt_msgs[1]["content"]
        gold = r["messages"][2]["content"]
        ft_answer = generate(ft_model, tokenizer, prompt_msgs)
        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print(f"GOLD:       {gold}")
        print(f"FINE-TUNED: {ft_answer}")
        results.append({"question": question, "gold": gold, "fine_tuned": ft_answer})

    # ---------- Optional: baseline regression comparison ----------
    if args.compare_base:
        del ft_model
        torch.cuda.empty_cache()
        print("\nLoading base model for comparison...")
        base = load_model(base_model)
        b_loss, b_ppl = test_loss(base, tokenizer, test_rows)
        print(f"[base] test loss: {b_loss:.4f} | perplexity: {b_ppl:.2f}")
        for r in results:
            r["base"] = generate(base, tokenizer, [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": r["question"]},
            ])

    out_path = ROOT / cfg["training"]["output_dir"] / "eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"test_loss": loss, "perplexity": ppl, "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"\nEval report saved to {out_path}")


if __name__ == "__main__":
    main()
