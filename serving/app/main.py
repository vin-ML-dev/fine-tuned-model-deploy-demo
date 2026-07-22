"""
NovaBot API - FastAPI wrapper around the model engine.

Phase 4: validation, streaming, timeouts, health probes, structured logs
Phase 6 additions:
  - Redis exact-match response cache (step 25): repeat questions answered
    in ~ms without calling the GPU engine (faster + cheaper)
  - Redis-backed rate limiting: correct across multiple replicas
    (falls back to in-memory when Redis is unavailable)
  - Backpressure / request queueing (step 24): a concurrency semaphore
    caps in-flight engine calls; beyond the queue limit -> 503 busy
  - A/B testing (step 27): deterministic traffic split between model
    variant A and variant B, tagged in responses and logs

Environment variables:
    ENGINE_BASE_URL     default http://localhost:8001/v1
    ENGINE_API_KEY      optional (RunPod serverless)
    MODEL_NAME          variant A model
    ENGINE_B_BASE_URL   optional - variant B engine (defaults to A's engine)
    MODEL_B_NAME        optional - variant B model (enables A/B when set)
    AB_SPLIT_PERCENT    0-100, % of traffic to variant B (default 0)
    REDIS_URL           e.g. redis://redis:6379/0 (empty = in-memory fallbacks)
    CACHE_TTL_S         default 3600
    MAX_CONCURRENCY     max parallel engine calls per replica (default 8)
    QUEUE_TIMEOUT_S     max seconds a request waits for a slot (default 10)
    REQUEST_TIMEOUT_S   engine call timeout (default 90)
    RATE_LIMIT_RPM      default 60
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENGINE_BASE_URL = os.getenv("ENGINE_BASE_URL", "http://localhost:8001/v1")
ENGINE_API_KEY = os.getenv("ENGINE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "vinmlops/technova-1.5b-instruct")

ENGINE_B_BASE_URL = os.getenv("ENGINE_B_BASE_URL", ENGINE_BASE_URL)
MODEL_B_NAME = os.getenv("MODEL_B_NAME", "")
AB_SPLIT_PERCENT = int(os.getenv("AB_SPLIT_PERCENT", "0"))

REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S", "3600"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
QUEUE_TIMEOUT_S = float(os.getenv("QUEUE_TIMEOUT_S", "10"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "90"))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))

SYSTEM_PROMPT = (
    "You are NovaBot, the official AI assistant of TechNova Solutions Pvt. Ltd. "
    "Answer questions about the company, its products, and its policies "
    "accurately and concisely, based on official company information. "
    "If a question is outside company matters, politely say you can only "
    "help with TechNova-related topics."
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("novabot-api")

app = FastAPI(title="NovaBot API", version="3.0.0")

# ---------------------------------------------------------------------------
# Phase 7 - Step 29: Prometheus metrics.
# Middleware counts EVERY response by path+status (incl. 4xx/5xx);
# chat-specific metrics add variant/cache dimensions; answer-length
# histogram is a cheap model-drift signal (step 31).
# ---------------------------------------------------------------------------
def _metric(cls, name, doc, labels, **kw):
    """Create a metric, or reuse the existing one if already registered
    (happens when the module is reloaded, e.g. in tests)."""
    try:
        return cls(name, doc, labels, **kw)
    except ValueError:
        from prometheus_client import REGISTRY
        return REGISTRY._names_to_collectors[name]


HTTP_REQUESTS = _metric(Counter, "novabot_http_requests_total",
                        "All HTTP responses", ["path", "method", "status"])
CHAT_REQUESTS = _metric(Counter, "novabot_chat_requests_total",
                        "Chat requests by outcome", ["variant", "cached", "status"])
CHAT_LATENCY = _metric(Histogram, "novabot_chat_latency_seconds",
                       "Chat end-to-end latency", ["variant", "cached"],
                       buckets=[0.01, 0.05, 0.25, 0.5, 1, 2, 5, 10, 30, 60])
ANSWER_CHARS = _metric(Histogram, "novabot_answer_chars",
                       "Answer length in characters (drift signal)", ["variant"],
                       buckets=[50, 100, 200, 400, 800, 1600])
FEEDBACK = _metric(Counter, "novabot_feedback_total",
                   "User feedback by rating", ["rating", "variant"])


@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path not in ("/metrics",):          # do not count scrapes
        HTTP_REQUESTS.labels(path=path, method=request.method,
                             status=str(response.status_code)).inc()
    return response

# ---------------------------------------------------------------------------
# Redis (optional). All Redis features degrade gracefully to local fallbacks
# so the app works identically in unit tests / no-Redis environments.
# ---------------------------------------------------------------------------
redis_client = None
if REDIS_URL:
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(
            REDIS_URL, decode_responses=True,
            socket_connect_timeout=2, socket_timeout=2,
        )
    except Exception as e:  # import or config error
        log.info(json.dumps({"event": "redis_init_failed", "error": str(e)}))
        redis_client = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default=[], max_length=20)
    user_id: str | None = Field(default=None, max_length=100,
                                description="Stable id for consistent A/B assignment")
    max_tokens: int = Field(default=250, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    stream: bool = False


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    model: str
    variant: str
    cached: bool
    latency_ms: int
    usage: dict | None = None


# ---------------------------------------------------------------------------
# Rate limiting: Redis fixed-window when available, in-memory sliding window
# fallback otherwise (single-replica only).
# ---------------------------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)


def _check_rate_limit_local(ip: str):
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    window.append(now)


async def check_rate_limit(ip: str):
    if redis_client is None:
        return _check_rate_limit_local(ip)
    try:
        key = f"rl:{ip}:{int(time.time() // 60)}"   # fixed 60s window bucket
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, 90)
        if count > RATE_LIMIT_RPM:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    except HTTPException:
        raise
    except Exception:
        _check_rate_limit_local(ip)   # Redis down -> degrade, don't fail requests


# ---------------------------------------------------------------------------
# A/B variant assignment (step 27).
# Deterministic: same user_id (or IP) always gets the same variant, so a
# user's experience is consistent and results are analyzable.
# ---------------------------------------------------------------------------
def assign_variant(stable_id: str) -> tuple[str, str, str]:
    """Returns (variant_label, model_name, engine_base_url)."""
    if MODEL_B_NAME and AB_SPLIT_PERCENT > 0:
        bucket = int(hashlib.sha256(stable_id.encode()).hexdigest(), 16) % 100
        if bucket < AB_SPLIT_PERCENT:
            return "B", MODEL_B_NAME, ENGINE_B_BASE_URL
    return "A", MODEL_NAME, ENGINE_BASE_URL


# ---------------------------------------------------------------------------
# Response cache (step 25). Exact-match on (variant, model, full message list,
# max_tokens). Only deterministic (temperature==0) non-streaming requests are
# cached - sampled outputs vary by design, so caching them would be wrong.
# ---------------------------------------------------------------------------
def cache_key(model: str, messages: list[dict], max_tokens: int) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "mt": max_tokens},
                      sort_keys=True, ensure_ascii=False)
    return "cache:" + hashlib.sha256(blob.encode()).hexdigest()


async def cache_get(key: str):
    if redis_client is None:
        return None
    try:
        val = await redis_client.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


async def cache_set(key: str, value: dict):
    if redis_client is None:
        return
    try:
        await redis_client.set(key, json.dumps(value, ensure_ascii=False), ex=CACHE_TTL_S)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Backpressure (step 24): at most MAX_CONCURRENCY engine calls in flight per
# replica; waiting requests queue up to QUEUE_TIMEOUT_S, then get 503.
# Protects the engine from thundering herds and keeps latency predictable.
# ---------------------------------------------------------------------------
_engine_slots = asyncio.Semaphore(MAX_CONCURRENCY)


# ---------------------------------------------------------------------------
# Engine client
# ---------------------------------------------------------------------------
_headers = {"Authorization": f"Bearer {ENGINE_API_KEY}"} if ENGINE_API_KEY else {}
client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, headers=_headers)


def build_messages(req: ChatRequest) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [t.model_dump() for t in req.history]
    msgs.append({"role": "user", "content": req.message})
    return msgs


async def engine_chat(base_url: str, payload: dict) -> dict:
    try:
        r = await client.post(f"{base_url}/chat/completions", json=payload)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model engine timed out.")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Model engine unavailable.")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Engine error: {r.text[:300]}")
    return r.json()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    ip = request.client.host
    await check_rate_limit(ip)
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    stable_id = req.user_id or ip
    variant, model, base_url = assign_variant(stable_id)
    messages = build_messages(req)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }

    # ---- streaming path (never cached) ----
    if req.stream:
        async def sse():
            try:
                async with client.stream(
                    "POST", f"{base_url}/chat/completions", json=payload
                ) as r:
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            yield line + "\n\n"
            except httpx.HTTPError:
                yield 'data: {"error": "engine stream failed"}\n\n'
        return StreamingResponse(sse(), media_type="text/event-stream")

    # ---- cache lookup (deterministic requests only) ----
    cacheable = req.temperature == 0.0
    key = cache_key(model, messages, req.max_tokens) if cacheable else None
    if cacheable:
        hit = await cache_get(key)
        if hit:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            CHAT_REQUESTS.labels(variant=variant, cached="true", status="ok").inc()
            CHAT_LATENCY.labels(variant=variant, cached="true").observe(latency_ms / 1000)
            log.info(json.dumps({
                "request_id": request_id, "variant": variant, "cached": True,
                "latency_ms": latency_ms, "status": "ok",
            }))
            return ChatResponse(
                request_id=request_id, answer=hit["answer"], model=model,
                variant=variant, cached=True, latency_ms=latency_ms,
                usage=hit.get("usage"),
            )

    # ---- backpressure: wait for an engine slot, or 503 when saturated ----
    try:
        await asyncio.wait_for(_engine_slots.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        CHAT_REQUESTS.labels(variant=variant, cached="false", status="busy").inc()
        raise HTTPException(status_code=503,
                            detail="Server busy. Please retry shortly.")
    try:
        data = await engine_chat(base_url, payload)
    finally:
        _engine_slots.release()

    answer = data["choices"][0]["message"]["content"].strip()
    latency_ms = int((time.perf_counter() - t0) * 1000)
    CHAT_REQUESTS.labels(variant=variant, cached="false", status="ok").inc()
    CHAT_LATENCY.labels(variant=variant, cached="false").observe(latency_ms / 1000)  #convert in seconds
    ANSWER_CHARS.labels(variant=variant).observe(len(answer))

    if cacheable:
        await cache_set(key, {"answer": answer, "usage": data.get("usage")})

    log.info(json.dumps({
        "request_id": request_id, "variant": variant, "cached": False,
        "latency_ms": latency_ms, "prompt_chars": len(req.message),
        "completion_chars": len(answer), "status": "ok",
    }))

    return ChatResponse(
        request_id=request_id, answer=answer, model=data.get("model", model),
        variant=variant, cached=False, latency_ms=latency_ms,
        usage=data.get("usage"),
    )


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: str = Field(pattern="^(up|down)$")
    variant: str = Field(default="A", pattern="^(A|B)$")
    comment: str | None = Field(default=None, max_length=1000)


@app.post("/v1/feedback")
async def feedback(fb: FeedbackRequest):
    """Step 32: user feedback loop. Counted in metrics + logged as JSON;
    these logs are the raw material for the Phase 8 retraining dataset."""
    FEEDBACK.labels(rating=fb.rating, variant=fb.variant).inc()
    log.info(json.dumps({
        "event": "feedback", "request_id": fb.request_id,
        "rating": fb.rating, "variant": fb.variant,
        "comment": fb.comment or "",
    }))
    return {"status": "recorded"}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz():
    """Liveness: is THIS process alive? (K8s livenessProbe)"""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness: engine reachable? Redis state reported but non-fatal."""
    redis_state = "disabled"
    if redis_client is not None:
        try:
            await redis_client.ping()
            redis_state = "up"
        except Exception:
            redis_state = "down"   # degraded but still serving (fallbacks active)
    try:
        r = await client.get(f"{ENGINE_BASE_URL}/models", timeout=5)
        if r.status_code == 200:
            return {"status": "ready", "engine": "up", "redis": redis_state}
    except httpx.HTTPError:
        pass
    raise HTTPException(status_code=503, detail="Engine not ready")


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
    if redis_client is not None:
        try:
            await redis_client.aclose()
        except Exception:
            pass
