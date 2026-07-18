"""
Phase 3 - Step 11a: Merge the LoRA adapter into the base model.

Why merge?
- vLLM serves a single merged model faster/simpler than base+adapter
- GGUF conversion (llama.cpp) requires a merged HF model

IMPORTANT: We do NOT load the base model in 4-bit here. Merging into
quantized weights is lossy. We load the base in bf16 (or fp16), apply the
adapter, merge, and save a full-precision model (~3 GB for 1.5B).
This needs ~8 GB RAM/VRAM - fine on your RunPod 3090, also runs on CPU.

Usage:
    python scripts/merge_adapter.py
    python scripts/merge_adapter.py --dtype float16   # if bf16 unsupported

Output: outputs/merged-technova-1.5b/
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/training_config.yaml"))
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--out", default=str(ROOT / "outputs/merged-technova-1.5b"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base_model = cfg["model"]["base_model"]
    adapter = ROOT / cfg["training"]["output_dir"] / "final_adapter"
    assert adapter.exists(), f"Adapter not found at {adapter}. Train first."

    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model {base_model} in {args.dtype} on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Applying adapter from {adapter}...")
    model = PeftModel.from_pretrained(model, str(adapter))

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # sanity check: one generation with the merged model
    msgs = [
        {"role": "system", "content": "You are NovaBot, the official AI assistant of TechNova Solutions Pvt. Ltd."},
        {"role": "user", "content": "What is the refund policy for monthly plans?"},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\nSanity check generation:\n", answer.strip())

    meta = {"base_model": base_model, "adapter": str(adapter), "dtype": args.dtype}
    with open(out_dir / "merge_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMerged model saved to: {out_dir}")


if __name__ == "__main__":
    main()
