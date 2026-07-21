# Project Phases — Status & Reference

**Project:** TechNova company chatbot — fine-tuned LLM, productionized end to end.
**Model:** `Qwen/Qwen2.5-1.5B-Instruct` + QLoRA → merged → published on HF Hub.
**Registry:** `vinmlops/technova-1.5b-instruct` (safetensors, for vLLM) · `vinmlops/technova-1.5b-instruct-GGUF` (Q4_K_M + bf16, for CPU).
**Constraint learned:** RunPod pods cannot run Docker/Kubernetes inside them — see the environment column.

---

## Phase status overview

| Phase | Roadmap steps | Status | Practice environment |
|---|---|---|---|
| 1. Problem & Data | 1–4 | ✅ Done | Anywhere (no GPU) |
| 2. Fine-Tuning (QLoRA) | 5–10 | ✅ Done | RunPod RTX 3090 |
| 3. Optimization for Serving | 11–13 | ✅ Done | RunPod RTX 3090 |
| 4. Serving Infrastructure | 14–18 | ✅ Built | RunPod (bare processes) / Docker on laptop |
| 5. Deployment (K8s + CI/CD) | 19–23 | ✅ Done | Laptop (kind/minikube) + GitHub Actions + GHCR |
| 6. Real-Time Prod Features | 24–27 | ✅ Built | Same K8s cluster as Phase 5 |
| 7. Observability & Reliability | 28–32 | ⏳ Pending | Same K8s cluster as Phase 5 |
| 8. Lifecycle (retrain, MLflow) | 33–36 | ⏳ Pending | RunPod (training) + K8s (deploy) |

---

## Phase 1 — Problem & Data ✅

| Step | What was done | File(s) |
|---|---|---|
| 1. Define task & metrics | Company QA chatbot "NovaBot"; success = low test loss + factually correct answers; SLO target p95 < 2s | README |
| 2. Collect data | Fictional company knowledge base with 14 sections (overview, mission, products, hours, support, refund, privacy, retention, handbook, leave, conduct, security, travel, performance) | `data/raw/company_knowledge_base.txt` |
| 3. Build dataset | ~90 curated question groups + paraphrase augmentation + out-of-scope refusal examples → 380 chat-format examples | `scripts/generate_dataset.py` |
| 4. Preprocess & split | Whitespace cleaning, question-level dedup, stratified 85/10/5 split per section, stats report for data versioning | `scripts/preprocess.py` |

**Result:** train 315 / val 34 / test 31 examples in `data/processed/`.

## Phase 2 — Fine-Tuning ✅

| Step | What was done | File(s) |
|---|---|---|
| 5. Choose model & method | Qwen2.5-1.5B-Instruct + QLoRA (4-bit NF4) + PEFT LoRA (r=16, all attn+MLP proj) | `configs/training_config.yaml` |
| 6. Experiment tracking | TensorBoard (MLflow upgrade planned in Phase 8) | config `report_to` |
| 7. Training script | Config-driven SFTTrainer; bf16 on Ampere (fp16 on T4); gradient checkpointing; paged 8-bit AdamW | `scripts/train_qlora.py` |
| 8. Train w/ checkpointing | Per-epoch checkpoints, early stopping (patience 2), load-best-at-end | same |
| 9. Evaluate | Test loss + perplexity, sample generations vs gold, `--compare_base` regression check | `scripts/evaluate.py` |
| 10. Model registry | Adapter + `model_card.json`; then pushed to HF Hub (registry of record) | `outputs/`, HF Hub |

**Known issue logged:** some fact-mixing in answers (trial vs refund, monthly vs annual). Fix scheduled for Phase 8 retrain: contrastive examples + 8 epochs + lr 1.5e-4 + patience 3.

## Phase 3 — Optimization for Serving ✅

| Step | What was done | File(s) |
|---|---|---|
| 11. Merge & convert | Adapter merged into base in **full bf16** (never merge into 4-bit); HF → GGUF bf16 via llama.cpp (hardened script: apt update, latest deps, sanity checks, absolute paths) | `scripts/merge_adapter.py`, `scripts/convert_to_gguf.sh` |
| 12. Quantize | bf16 GGUF → Q4_K_M (~1 GB) | same |
| 13. Benchmark | p50/p95 latency, tokens/sec, peak GPU memory → baseline vs vLLM later | `scripts/benchmark.py` → `benchmark_baseline.json` |

**Lessons captured:** dtype consistency end-to-end (bf16 train → merge → GGUF); converter errors were path + stale-checkout issues, not tokenizer issues.

## Phase 4 — Serving Infrastructure ✅

| Step | What was done | File(s) |
|---|---|---|
| 14. Serving engine | vLLM (GPU, serves safetensors repo) and llama.cpp server (CPU, serves GGUF) — both OpenAI-compatible | compose files |
| 15. Inference API | FastAPI wrapper: Pydantic validation, server-side NovaBot system prompt (prevents raw-prompt bug), multi-turn history | `serving/app/main.py` |
| 16. Real-time features | SSE streaming, timeout handling (504), engine-down handling (503), per-IP sliding-window rate limiting (429) | same |
| 17. Containerize | Slim non-root API image; engine images pulled (vllm/vllm-openai, ggml llama.cpp:server); model weights pulled from HF Hub at start — never baked into images | `serving/Dockerfile.api`, `serving/docker-compose.{gpu,cpu}.yml` |
| 18. Health endpoints | `/healthz` (liveness) vs `/readyz` (readiness, pings engine) → map 1:1 to K8s probes | `serving/app/main.py` |
| — RunPod adaptation | No Docker on pods → bare-process launcher (vLLM + uvicorn, logs, readiness wait, stop command, proxy-URL exposure) | `serving/run_on_runpod.sh` |
| — Smoke tests | health/readiness/validation/QA/multi-turn — becomes the CI test suite in Phase 5 | `serving/test_api.py` |

## Phase 5 — Deployment (K8s + CI/CD) ⏳ NEXT

| Step | Plan |
|---|---|
| 19. Infrastructure | minikube or kind on laptop/VPS (CPU GGUF path — no GPU needed); RunPod pods can't host K8s |
| 20. Autoscaling | HPA on CPU/memory (queue-depth based later) |
| 21. CI/CD | GitHub Actions: lint + smoke tests → build image → push registry → deploy |
| 22. Rollout strategy | Rolling update first, then blue-green/canary |
| 23. Gateway & auth | Ingress + API-key auth |

## Phase 6 — Real-Time Production Features ⏳

| Step | Plan |
|---|---|
| 24. Request queueing | Engine-level (vLLM continuous batching) + API backpressure |
| 25. Caching | Redis exact-match cache; rate limiter moves from in-memory to Redis |
| 26. Feature store | Discussed conceptually (not needed for this chatbot) |
| 27. A/B testing | Two model versions behind weighted routing |

## Phase 7 — Observability & Reliability ⏳

| Step | Plan |
|---|---|
| 28. Logging | Structured JSON logs (already emitted by API) → aggregation |
| 29. Metrics & dashboards | Prometheus `/metrics` on API + vLLM metrics → Grafana (p50/p95/p99, error rate, GPU) |
| 30. Alerting | Alertmanager rules (error rate, latency, engine down) |
| 31. Model monitoring | Drift signals: out-of-scope rate, answer-length shifts, thumbs-down feedback |
| 32. Feedback loop | Feedback endpoint → data for retraining |

## Phase 8 — Lifecycle ⏳

| Step | Plan |
|---|---|
| 33. Retraining pipeline | **Includes the planned quality retrain**: contrastive dataset v2, 8 epochs, lr 1.5e-4, patience 3; automated trigger later; **MLflow** replaces TensorBoard for tracking + registry |
| 34. Versioning & rollback | HF Hub revisions/tags; K8s rollback via image tags |
| 35. Cost monitoring | Pod-hour tracking, serverless vs pod tradeoffs |
| 36. Docs & runbooks | Incident runbooks, on-call notes |

---

## Repo file map

```
company-chatbot-production/
├── README.md                     # step-by-step guide, all phases
├── PHASES.md                     # this file
├── requirements.txt              # training deps (Phases 1–2)
├── configs/training_config.yaml  # all hyperparameters
├── data/raw/company_knowledge_base.txt
├── scripts/
│   ├── generate_dataset.py       # Phase 1
│   ├── preprocess.py             # Phase 1
│   ├── train_qlora.py            # Phase 2
│   ├── evaluate.py               # Phase 2
│   ├── merge_adapter.py          # Phase 3
│   ├── convert_to_gguf.sh        # Phase 3 (hardened)
│   └── benchmark.py              # Phase 3
└── serving/                      # Phase 4
    ├── app/main.py               # FastAPI wrapper
    ├── requirements-api.txt
    ├── Dockerfile.api
    ├── docker-compose.gpu.yml    # vLLM path
    ├── docker-compose.cpu.yml    # llama.cpp path
    ├── run_on_runpod.sh          # bare-process launcher for RunPod
    └── test_api.py               # smoke tests (future CI suite)
```

## Key decisions log

| Decision | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | T4/3090-friendly, non-gated, cheap to serve |
| Precision | bf16 (Ampere) / fp16 (T4) | T4 has no bf16; keep dtype consistent train→merge→GGUF |
| Merge precision | Full bf16, never 4-bit | Merging into quantized weights is lossy |
| GGUF quant | Q4_K_M | Standard best quality/size tradeoff |
| Registry | HF Hub (2 repos) | Free, versioned, pullable from Docker/K8s — no weights in images |
| API pattern | Engine-agnostic sidecar | One FastAPI wrapper for both vLLM and llama.cpp |
| Inference rule | Always chat template + training system prompt, enforced server-side | Raw prompts break fine-tuned behavior (verified painfully) |
| Decoding | Greedy (temperature 0) | Deterministic facts for a policy bot |
| K8s environment | Laptop/VPS, not RunPod | RunPod pods can't run Docker/K8s inside |
