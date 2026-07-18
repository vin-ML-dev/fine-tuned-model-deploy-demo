"""
Phase 4: Smoke tests for the NovaBot API.
Run against a live stack (docker compose up first):

    python serving/test_api.py
    python serving/test_api.py --base http://localhost:8000
"""

import argparse
import sys

import httpx

QUESTIONS = [
    "What is the refund policy for monthly subscriptions?",
    "How many paid leave days do employees get per year?",
    "What is the password policy?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    failures = 0

    with httpx.Client(timeout=120) as c:
        # 1. liveness
        r = c.get(f"{base}/healthz")
        print(f"/healthz -> {r.status_code} {r.json()}")
        assert r.status_code == 200

        # 2. readiness (engine up?)
        r = c.get(f"{base}/readyz")
        print(f"/readyz  -> {r.status_code} {r.json() if r.status_code==200 else r.text}")
        if r.status_code != 200:
            sys.exit("Engine not ready - is the engine container healthy?")

        # 3. validation: empty message must be rejected (422)
        r = c.post(f"{base}/v1/chat", json={"message": ""})
        print(f"validation test (empty msg) -> {r.status_code} (expect 422)")
        failures += r.status_code != 422

        # 4. real questions
        for q in QUESTIONS:
            r = c.post(f"{base}/v1/chat", json={"message": q})
            if r.status_code != 200:
                print(f"FAIL [{r.status_code}] {q}: {r.text[:200]}")
                failures += 1
                continue
            d = r.json()
            print(f"\nQ: {q}\nA: {d['answer']}\n[{d['latency_ms']} ms, id={d['request_id']}]")

        # 5. multi-turn history
        r = c.post(f"{base}/v1/chat", json={
            "message": "And what about annual plans?",
            "history": [
                {"role": "user", "content": "What is the refund policy for monthly subscriptions?"},
                {"role": "assistant", "content": "Full refund within 7 days of the billing date."},
            ],
        })
        print(f"\nmulti-turn -> {r.status_code}")
        if r.status_code == 200:
            print("A:", r.json()["answer"])

    print(f"\n{'ALL SMOKE TESTS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
