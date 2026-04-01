"""NLU: map symptoms to specialty IDs from a fixed catalog (IDs must match DB/Qdrant)."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from medicalink_ai.config import Settings
from medicalink_ai.gemini_llm import generate_json_with_gemini

logger = logging.getLogger(__name__)

SYSTEM = """You classify the patient's Vietnamese description into medical specialties.
Only use ids that appear in the provided list. Never invent ids.
If the case is vague or could belong to very different departments, return an empty specialty_ids array.
Return a single JSON object:
{"specialty_ids":["id1",...], "note":"1–2 short Vietnamese sentences for the patient"}
- At most 4 ids, most likely matches first.
"""


async def suggest_specialties_from_catalog(
    *,
    symptoms: str,
    catalog: list[dict[str, str]],
    settings: Settings,
    openai: AsyncOpenAI,
) -> dict[str, Any]:
    allowed = {str(x.get("id", "")).strip() for x in catalog if x.get("id")}
    allowed.discard("")
    if not allowed:
        return {"specialty_ids": [], "note": "No specialties in catalog."}

    lines: list[str] = []
    for x in catalog[:120]:
        i = str(x.get("id", "")).strip()
        n = str(x.get("name", "")).strip()
        if i and n:
            lines.append(json.dumps({"id": i, "name": n}, ensure_ascii=False))
    user_msg = (
        f"Patient description:\n{symptoms.strip()}\n\n"
        f"Specialties (choose ids only from this list):\n"
        + "\n".join(lines)
    )

    prov = (settings.llm_provider or "openai").strip().lower()
    temp = float(settings.rag_llm_temperature)
    try:
        if prov == "gemini":
            raw = await generate_json_with_gemini(
                api_key=settings.google_genai_api_key.strip(),
                model=settings.google_genai_model.strip(),
                system_instruction=SYSTEM,
                user_content=user_msg,
                timeout_ms=max(5_000, int(settings.google_genai_timeout_ms)),
                temperature=temp,
            )
        else:
            completion = await openai.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=temp,
            )
            raw = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning("specialty suggestion failed: %s", e)
        return {
            "specialty_ids": [],
            "note": "Could not classify automatically; please select specialties yourself.",
        }

    if not isinstance(parsed, dict):
        return {"specialty_ids": [], "note": ""}

    raw_ids = parsed.get("specialty_ids")
    out_ids: list[str] = []
    if isinstance(raw_ids, list):
        for x in raw_ids[:6]:
            sid = str(x).strip()
            if sid in allowed and sid not in out_ids:
                out_ids.append(sid)
    out_ids = out_ids[:4]

    note = str(parsed.get("note") or "").strip()
    if not note:
        note = (
            "Suggested specialties based on your description."
            if out_ids
            else "Consider choosing a specialty manually."
        )

    return {"specialty_ids": out_ids, "note": note}
