"""NLU: chọn chuyên khoa từ catalog cố định (id khớp DB/Qdrant)."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from medicalink_ai.config import Settings
from medicalink_ai.gemini_llm import generate_json_with_gemini

logger = logging.getLogger(__name__)

SYSTEM = """Bạn là trợ lý phân loại triệu chứng sang chuyên khoa (Việt Nam).
Chỉ được chọn id nằm trong danh sách được cung cấp. Không bịa id.
Nếu triệu chứng mơ hồ hoặc có thể thuộc nhiều khoa quá khác nhau, trả mảng rỗng.
Trả về JSON duy nhất:
{"specialty_ids":["id1",...], "note":"1-2 câu tiếng Việt giải thích ngắn cho bệnh nhân"}
- Tối đa 4 id, ưu tiên các khoa khả dĩ nhất theo triệu chứng.
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
        return {"specialty_ids": [], "note": "Danh sách chuyên khoa trống."}

    lines: list[str] = []
    for x in catalog[:120]:
        i = str(x.get("id", "")).strip()
        n = str(x.get("name", "")).strip()
        if i and n:
            lines.append(json.dumps({"id": i, "name": n}, ensure_ascii=False))
    user_msg = (
        f"Triệu chứng / mô tả:\n{symptoms.strip()}\n\n"
        f"Danh sách chuyên khoa (chỉ chọn id trong list):\n"
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
        logger.warning("specialty suggestion LLM/JSON lỗi: %s", e)
        return {
            "specialty_ids": [],
            "note": "Không phân loại được tự động; vui lòng tự chọn chuyên khoa.",
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
            "Đã gợi ý chuyên khoa dựa trên mô tả của bạn."
            if out_ids
            else "Nên để bạn hoặc nhân viên y tế chọn chuyên khoa phù hợp hơn."
        )

    return {"specialty_ids": out_ids, "note": note}
