"""Gọi Gemini (google-genai) cho bước sinh JSON trong RAG — chạy sync SDK trong thread pool."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def generate_json_with_gemini(
    *,
    api_key: str,
    model: str,
    system_instruction: str,
    user_content: str,
    timeout_ms: int,
    temperature: float = 0.2,
) -> str:
    def _sync_call() -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        resp = client.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )
        text = getattr(resp, "text", None) or ""
        if not text and getattr(resp, "candidates", None):
            try:
                c0 = resp.candidates[0]
                parts = c0.content.parts
                text = "".join(getattr(p, "text", "") or "" for p in parts)
            except (IndexError, AttributeError) as e:
                logger.debug("Gemini response fallback empty: %s", e)
        return text or "{}"

    return await asyncio.to_thread(_sync_call)
