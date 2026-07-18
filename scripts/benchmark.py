"""
Phase 3 - Step 13: Benchmark the merged model - latency & throughput.

Measures, over N runs of realistic company questions:
  - Time to first token (TTFT) approximation
  - End-to-end latency: p50 / p95
  - Generation throughput (tokens/sec)
  - Peak GPU memory

These numbers become your baseline SLO reference before vLLM (Phase 4),
so you can prove vLLM's improvement later.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --runs 20 --max_new_tokens 200
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    "What is TechNova's refund policy for monthly subscriptions?",
    "How many paid leave days do employees get per year?",
    "What are the customer support response time targets?",
    "What is the password policy at TechNova?",
    "Tell me about the NovaCloud product and its pricing.",
    "What class can I fly for international business travel?",
    "How long is customer data retained after account closure?",
    "When do performance reviews happen and what is the rating scale?",
]

SYSTEM = "You are NovaBot, the official AI assistant of TechNova Solutions Pvt. Ltd."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "outputs/merged-technova-1.5b"))
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    print(f"Loading {args.model} ({dtype}) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    )
    model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- warmup (first run includes CUDA kernel compilation - not counted) ----
    def run_once(question, max_new):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        n_out = out.shape[1] - n_in
        return elapsed, n_out

    print("Warmup...")
    run_once(QUESTIONS[0], 20)

    # ---- timed runs ----
    latencies, tps_list = [], []
    for i in range(args.runs):
        q = QUESTIONS[i % len(QUESTIONS)]
        elapsed, n_out = run_once(q, args.max_new_tokens)
        latencies.append(elapsed)
        tps_list.append(n_out / elapsed)
        print(f"run {i+1:>2}/{args.runs}: {elapsed:.2f}s | {n_out} tokens | {n_out/elapsed:.1f} tok/s")

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    avg_tps = statistics.mean(tps_list)

    report = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "runs": args.runs,
        "max_new_tokens": args.max_new_tokens,
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "throughput_tok_per_s_avg": round(avg_tps, 1),
    }
    if device == "cuda":
        report["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    print("\n===== BENCHMARK SUMMARY =====")
    for k, v in report.items():
        print(f"{k:>28}: {v}")

    out_path = ROOT / "outputs" / "benchmark_baseline.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path} (baseline to compare against vLLM in Phase 4)")


if __name__ == "__main__":
    main()
