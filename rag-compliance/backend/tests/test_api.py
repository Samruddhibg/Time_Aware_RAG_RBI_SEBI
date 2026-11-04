"""
tests/test_api.py — API smoke tests using httpx TestClient.
"""
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_docs():
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_ask_validation_short_query():
    """Short query should fail validation."""
    resp = client.post("/api/ask", json={"question": "hi"})
    assert resp.status_code == 422


def test_ask_validation_missing_question():
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ask_sse_stream():
    """Test that /ask returns SSE format."""
    mock_chunks = [
        {
            "chunk_id": 1,
            "document_id": "RBI-TEST-001",
            "circular_no": "RBI/2024/01",
            "title": "Test Circular",
            "section": "Section 1",
            "text": "The KYC limit is Rs 50,000",
            "issue_date": "2024-01-01",
            "status": "active",
            "similarity": 0.9,
            "snippet": "The KYC limit is Rs 50,000",
        }
    ]

    async def mock_rag_stream(*args, **kwargs):
        yield {"type": "sources", "data": mock_chunks}
        yield {"type": "token", "data": "The "}
        yield {"type": "token", "data": "KYC limit is Rs 50,000 [1]."}
        yield {"type": "done", "data": {"query_id": 1, "refused": False,
                                         "retrieval_ms": 50, "total_ms": 100}}

    with patch("app.routers.ask.rag_stream", side_effect=mock_rag_stream):
        with patch.object(app.state, "pool", MagicMock()):
            resp = client.post(
                "/api/ask",
                json={"question": "What is the KYC cash transaction limit?"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            events = []
            for line in resp.text.splitlines():
                if line.startswith("data:") and not line.endswith("[DONE]"):
                    events.append(json.loads(line[5:].strip()))
            types = [e["type"] for e in events]
            assert "sources" in types
            assert "token" in types
            assert "done" in types
