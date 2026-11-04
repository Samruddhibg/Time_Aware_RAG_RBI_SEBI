"""
services/llm.py — OpenAI-compatible streaming client (works with Ollama, vLLM, OpenAI).
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
from loguru import logger

from app.core.config import settings


async def stream_completion(
    system_prompt: str,
    user_prompt: str,
) -> AsyncIterator[str]:
    """
    Async generator that yields token strings as they arrive from the LLM.
    Supports OpenAI-compatible /v1/chat/completions SSE format.
    """
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(f"LLM error {resp.status_code}: {body.decode()[:500]}")
                yield f"[LLM error {resp.status_code}]"
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def complete(system_prompt: str, user_prompt: str) -> str:
    """Non-streaming version — accumulates the full response."""
    tokens: list[str] = []
    async for token in stream_completion(system_prompt, user_prompt):
        tokens.append(token)
    return "".join(tokens)
