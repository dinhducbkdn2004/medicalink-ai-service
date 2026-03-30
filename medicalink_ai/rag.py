from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from medicalink_ai.vector_store import DoctorVectorStore

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là trợ lý y tế chỉ được trả lời dựa trên danh sách bác sĩ được cấp (ngữ cảnh).
Nhiệm vụ: gợi ý các bác sĩ phù hợp nhất với mô tả triệu chứng của bệnh nhân.
Quy tắc:
- Chỉ chọn bác sĩ có trong ngữ cảnh; không bịa bác sĩ hoặc chuyên khoa.
- Mỗi bác sĩ phải dùng đúng doctor_id từ ngữ cảnh (UUID chuỗi).
- Trả về JSON duy nhất với cấu trúc:
{"recommendations":[{"doctor_id":"...","reason":"... ngắn gọn tiếng Việt..."}]}
- reason tối đa 2 câu, thân thiện, không chẩn đoán chắc chắn thay bác sĩ.
"""


class DoctorRagService:
    def __init__(
        self,
        store: DoctorVectorStore,
        openai: AsyncOpenAI,
        chat_model: str,
    ) -> None:
        self.store = store
        self.openai = openai
        self.chat_model = chat_model

    async def recommend(self, symptoms: str, top_k: int) -> dict[str, Any]:
        retrieve_n = min(max(top_k * 4, top_k + 5), 25)
        candidates = await self.store.search_active(symptoms, limit=retrieve_n)
        if not candidates:
            return {
                "recommendations": [],
                "message": "Chưa có hồ sơ bác sĩ để gợi ý. Hãy chạy đồng bộ knowledge base hoặc kiểm tra Qdrant.",
            }

        ctx_lines = []
        for c in candidates:
            ctx_lines.append(
                json.dumps(
                    {
                        "doctor_id": c.get("doctor_id"),
                        "full_name": c.get("full_name"),
                        "specialties": c.get("specialties_label"),
                        "score": c.get("score"),
                    },
                    ensure_ascii=False,
                )
            )
        context_block = "\n".join(ctx_lines)

        user_msg = f"Triệu chứng / mô tả của bệnh nhân:\n{symptoms}\n\nNgữ cảnh bác sĩ:\n{context_block}\n\nChọn tối đa {top_k} bác sĩ, xếp theo độ phù hợp."

        completion = await self.openai.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = completion.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM JSON parse failed, fallback regex")
            parsed = _fallback_parse(raw, candidates, top_k)

        recs = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(recs, list):
            recs = []

        allowed = {str(c["doctor_id"]) for c in candidates if c.get("doctor_id")}
        cleaned = []
        for r in recs[:top_k]:
            if not isinstance(r, dict):
                continue
            did = str(r.get("doctor_id") or "")
            if did not in allowed:
                continue
            reason = str(r.get("reason") or "").strip()
            cleaned.append({"doctor_id": did, "reason": reason})

        if len(cleaned) < top_k:
            cleaned = _fill_from_candidates(cleaned, candidates, top_k)

        return {"recommendations": cleaned[:top_k]}


def _fallback_parse(
    raw: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    out: list[dict[str, str]] = []
    for m in re.finditer(
        r'"doctor_id"\s*:\s*"([^"]+)"\s*,\s*"reason"\s*:\s*"([^"]*)"',
        raw,
    ):
        out.append({"doctor_id": m.group(1), "reason": m.group(2)})
        if len(out) >= top_k:
            break
    return {"recommendations": out}


def _fill_from_candidates(
    existing: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, str]]:
    have = {x["doctor_id"] for x in existing}
    for c in candidates:
        if len(existing) >= top_k:
            break
        did = str(c.get("doctor_id") or "")
        if not did or did in have:
            continue
        have.add(did)
        existing.append(
            {
                "doctor_id": did,
                "reason": "Gợi ý dựa trên độ tương đồng kỹ thuật với mô tả của bạn (không thay cho tư vấn y khoa).",
            }
        )
    return existing
