"""
Phase 4 - Steps 15-18: FastAPI wrapper around the model engine.

Architecture (sidecar pattern):
    client -> FastAPI (this app) -> engine (vLLM on GPU  OR  llama.cpp on CPU)

Both engines expose an OpenAI-compatible /v1/chat/completions API, so this
wrapper is engine-agnostic. It adds the production layer:
  - request validation (Pydantic)
  - the official NovaBot system prompt enforced server-side
  - response streaming (SSE) and non-streaming
  - timeouts + upstream error handling
  - simple in-memory rate limiting per client IP
  - /healthz (liveness) and /readyz (readiness: checks the engine)
  - structured JSON logs with request IDs and latency

Run locally:
    uvicorn serving.app.main:app --host 0.0.0.0 --port 8000

Environment variables:
    ENGINE_BASE_URL   default http://localhost:8001/v1   (vLLM or llama.cpp server)
    MODEL_NAME        default vinmlops/technova-1.5b-instruct
    REQUEST_TIMEOUT_S default 60
    RATE_LIMIT_RPM    default 60 (requests/min per IP)
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENGINE_BASE_URL = os.getenv("ENGINE_BASE_URL", "http://localhost:8001/v1")
ENGINE_API_KEY = os.getenv("ENGINE_API_KEY", "")   # e.g. RunPod API key for serverless endpoints
MODEL_NAME = os.getenv("MODEL_NAME", "vinmlops/technova-1.5b-instruct")
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "90"))  # allow serverless cold starts
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

app = FastAPI(title="NovaBot API", version="1.0.0")

# ---------------------------------------------------------------------------
# Schemas (request validation - step 15)
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000,
                         description="The user's question")
    history: list[ChatTurn] = Field(default=[], max_length=20,
                                    description="Prior turns, oldest first")
    max_tokens: int = Field(default=250, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    stream: bool = False


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    model: str
    latency_ms: int
    usage: dict | None = None


# ---------------------------------------------------------------------------
# Simple sliding-window rate limiter per IP (step 16)
# For multi-replica production this moves to Redis (Phase 6).
# ---------------------------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)


def check_rate_limit(ip: str):
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    window.append(now)


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


async def engine_chat(payload: dict) -> dict:
    try:
        r = await client.post(f"{ENGINE_BASE_URL}/chat/completions", json=payload)
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
    check_rate_limit(request.client.host)
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    payload = {
        "model": MODEL_NAME,
        "messages": build_messages(req),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }

    # ---- streaming path (SSE passthrough) ----
    if req.stream:
        async def sse():
            try:
                async with client.stream(
                    "POST", f"{ENGINE_BASE_URL}/chat/completions", json=payload
                ) as r:
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            yield line + "\n\n"
            except httpx.HTTPError:
                yield 'data: {"error": "engine stream failed"}\n\n'
        return StreamingResponse(sse(), media_type="text/event-stream")

    # ---- non-streaming path ----
    data = await engine_chat(payload)
    answer = data["choices"][0]["message"]["content"].strip()
    latency_ms = int((time.perf_counter() - t0) * 1000)

    log.info(json.dumps({
        "request_id": request_id,
        "path": "/v1/chat",
        "latency_ms": latency_ms,
        "prompt_chars": len(req.message),
        "completion_chars": len(answer),
        "status": "ok",
    }))

    return ChatResponse(
        request_id=request_id,
        answer=answer,
        model=data.get("model", MODEL_NAME),
        latency_ms=latency_ms,
        usage=data.get("usage"),
    )


@app.get("/healthz")
async def healthz():
    """Liveness: is THIS process alive? (K8s livenessProbe)"""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness: can we actually serve? Checks the engine. (K8s readinessProbe)"""
    try:
        r = await client.get(f"{ENGINE_BASE_URL}/models", timeout=5)
        if r.status_code == 200:
            return {"status": "ready", "engine": "up"}
    except httpx.HTTPError:
        pass
    raise HTTPException(status_code=503, detail="Engine not ready")


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
