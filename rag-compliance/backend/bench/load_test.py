"""
bench/load_test.py — Locust load test for /api/ask.

Usage:
    locust -f bench/load_test.py --headless -u 10 -r 2 --run-time 60s --host http://localhost:8000
"""
from locust import HttpUser, task, between
import json
import random

_QUERIES = [
    "What is the KYC cash transaction limit?",
    "What documents are needed for NRI account opening?",
    "What is the CRAR requirement for Indian banks?",
    "What is SEBI's insider trading definition?",
    "What are the priority sector lending targets for banks?",
    "How often must high-risk customers update KYC?",
    "What is the NPA classification timeline?",
    "What is the minimum net worth for SEBI-registered investment advisors?",
    "What are the NBFC categories defined by RBI?",
    "What is the LTV ratio for gold loans by NBFCs?",
]


class ComplianceUser(HttpUser):
    wait_time = between(1, 3)

    @task(10)
    def ask_current(self):
        """Ask a current-rules question (most common)."""
        question = random.choice(_QUERIES)
        with self.client.post(
            "/api/ask",
            json={"question": question, "historical": False},
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                # Consume SSE stream
                content = b""
                for chunk in resp.iter_content(chunk_size=512):
                    content += chunk
                if b'"type": "done"' in content or b'"type":"done"' in content:
                    resp.success()
                else:
                    resp.failure("No 'done' event in SSE stream")
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def ask_historical(self):
        """Historical mode query."""
        question = random.choice(_QUERIES)
        with self.client.post(
            "/api/ask",
            json={"question": question, "historical": True},
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/api/health")
