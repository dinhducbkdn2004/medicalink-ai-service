from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

from openai import AsyncOpenAI

from medicalink_ai.config import Settings
from medicalink_ai.eval_log import StepTimer, append_rag_eval, build_eval_record
from medicalink_ai.gemini_llm import generate_json_with_gemini
from medicalink_ai.rerank import RerankMode, query_seniority_intent, rerank_pipeline
from medicalink_ai.vector_store import DoctorVectorStore

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là trợ lý y tế chỉ được trả lời dựa trên danh sách bác sĩ được cấp (JSON: full_name, specialties, specialty_ids, locations, seniority_hint, retrieval_score).
Nhiệm vụ: gợi ý tối đa số bác sĩ được yêu cầu, xếp từ phù hợp nhất trở xuống.
Quy tắc bắt buộc:
- Chỉ dùng doctor_id có trong ngữ cảnh; không bịa hồ sơ.
- Ngữ cảnh có thể đã lọc theo chuyên khoa (specialty_ids) — nếu user nhắc khoa cụ thể, ưu tiên bác sĩ có specialties khớp trực tiếp.
- Ưu tiên bác sĩ có chuyên khoa / lĩnh vực khớp mô tả bệnh nhân. Không “ép” bác sĩ chuyên khoa xa trừ khi ngữ cảnh chứng minh liên quan.
- Nếu user nhấn mạnh “giỏi / lâu năm / kinh nghiệm”, ưu tiên bác sĩ có seniority_hint cao hơn khi chọn thứ tự.
- reason (1–2 câu tiếng Việt): nêu cụ thể — trích specialties (và locations nếu có); có thể nhắc học hàm/chức vụ nếu khớp triệu chứng. Tránh câu chung như “có thể hỗ trợ”. Không chẩn đoán bệnh thay bác sĩ.
- Trả về đúng một JSON: {"recommendations":[{"doctor_id":"...","reason":"..."}]}
"""


class DoctorRagService:
    def __init__(
        self,
        store: DoctorVectorStore,
        openai: AsyncOpenAI,
        settings: Settings,
    ) -> None:
        self.store = store
        self.openai = openai
        self.settings = settings

    async def recommend(
        self,
        symptoms: str,
        top_k: int,
        *,
        specialty_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        timer = StepTimer()
        top_k = max(1, min(top_k, 15))
        retrieve_limit = min(max(top_k * 6, 30), 80)
        prov = (self.settings.llm_provider or "openai").strip().lower()
        spec_filter = specialty_ids or None

        candidates, hybrid_used, legacy_col = await self.store.search_active(
            symptoms,
            limit=retrieve_limit,
            filter_specialty_ids=spec_filter,
        )
        if not candidates:
            out = {
                "recommendations": [],
                "message": "Chưa có hồ sơ bác sĩ để gợi ý. Hãy chạy đồng bộ knowledge base hoặc kiểm tra Qdrant.",
            }
            append_rag_eval(
                self.settings.rag_eval_log_path,
                build_eval_record(
                    query=symptoms,
                    top_k=top_k,
                    retrieved_ids=[],
                    recommended_ids=[],
                    hybrid_used=hybrid_used,
                    legacy_collection=legacy_col,
                    rerank_mode=str(self.settings.rag_rerank_mode),
                    latency_ms=timer.elapsed_ms(),
                    extra={
                        "empty": True,
                        "specialty_filter_n": len(spec_filter) if spec_filter else 0,
                    },
                ),
            )
            return out

        mode_raw = (self.settings.rag_rerank_mode or "lexical").strip().lower()
        if mode_raw not in ("none", "lexical", "flashrank"):
            mode_raw = "lexical"
        rmode = cast(RerankMode, mode_raw)

        ranked = rerank_pipeline(
            symptoms,
            candidates,
            mode=rmode,
            lexical_weight=float(self.settings.rag_rerank_lexical_weight),
            flashrank_model=self.settings.flashrank_model,
            flashrank_cache_dir=self.settings.flashrank_cache_dir or None,
            flashrank_pool=int(self.settings.rag_rerank_pool),
            seniority_boost_weight=float(self.settings.rag_seniority_boost_weight),
        )

        ctx_max = max(5, int(self.settings.retrieval_llm_context_max))
        for_llm = ranked[:ctx_max]

        ctx_lines = []
        for c in for_llm:
            loc = str(c.get("locations_label") or "").strip()
            sids = c.get("specialty_ids") or []
            if not isinstance(sids, list):
                sids = []
            ctx_lines.append(
                json.dumps(
                    {
                        "doctor_id": c.get("doctor_id"),
                        "full_name": c.get("full_name"),
                        "specialties": c.get("specialties_label"),
                        "specialty_ids": sids[:12],
                        "locations": loc or None,
                        "seniority_hint": round(float(c.get("seniority_score") or 0.0), 3),
                        "retrieval_score": c.get("score"),
                    },
                    ensure_ascii=False,
                )
            )
        context_block = "\n".join(ctx_lines)

        user_msg = f"Triệu chứng / mô tả của bệnh nhân:\n{symptoms}\n\nNgữ cảnh bác sĩ:\n{context_block}\n\nChọn tối đa {top_k} bác sĩ, xếp theo độ phù hợp."

        temp = float(self.settings.rag_llm_temperature)
        if prov == "gemini":
            raw = await generate_json_with_gemini(
                api_key=self.settings.google_genai_api_key.strip(),
                model=self.settings.google_genai_model.strip(),
                system_instruction=SYSTEM_PROMPT,
                user_content=user_msg,
                timeout_ms=max(5_000, int(self.settings.google_genai_timeout_ms)),
                temperature=temp,
            )
        else:
            completion = await self.openai.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=temp,
            )
            raw = completion.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM JSON parse failed, fallback regex")
            parsed = _fallback_parse(raw, for_llm, top_k)

        recs = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(recs, list):
            recs = []

        allowed = {str(c["doctor_id"]) for c in for_llm if c.get("doctor_id")}
        cleaned: list[dict[str, str]] = []
        for r in recs[: top_k * 3]:
            if not isinstance(r, dict):
                continue
            did = str(r.get("doctor_id") or "")
            if did not in allowed:
                continue
            reason = str(r.get("reason") or "").strip()
            cleaned.append({"doctor_id": did, "reason": reason})

        cleaned = _dedupe_preserve_order(cleaned)

        if not cleaned and for_llm:
            logger.warning("LLM không trả id hợp lệ — fallback thứ tự retrieval")
            cleaned = _fill_from_candidates([], for_llm, top_k)
        elif len(cleaned) < top_k:
            cleaned = _fill_from_candidates(cleaned, for_llm, top_k)

        _backfill_empty_reasons(cleaned, for_llm)

        rec_ids = [x["doctor_id"] for x in cleaned[:top_k]]
        ret_ids = [
            str(c.get("doctor_id")) for c in ranked if c.get("doctor_id")
        ]

        append_rag_eval(
            self.settings.rag_eval_log_path,
            build_eval_record(
                query=symptoms,
                top_k=top_k,
                retrieved_ids=ret_ids[:60],
                recommended_ids=rec_ids,
                hybrid_used=hybrid_used,
                legacy_collection=legacy_col,
                rerank_mode=mode_raw,
                latency_ms=timer.elapsed_ms(),
                extra={
                    "retrieve_n": len(candidates),
                    "after_rerank_n": len(ranked),
                    "llm_context_n": len(for_llm),
                    "llm_provider": prov,
                    "specialty_filter_n": len(spec_filter) if spec_filter else 0,
                    "seniority_intent": bool(query_seniority_intent(symptoms)),
                },
            ),
        )

        return {"recommendations": cleaned[:top_k]}


def _dedupe_preserve_order(recs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for r in recs:
        did = str(r.get("doctor_id") or "")
        if not did or did in seen:
            continue
        seen.add(did)
        out.append(r)
    return out


def _backfill_empty_reasons(
    recs: list[dict[str, str]],
    for_llm: list[dict[str, Any]],
) -> None:
    by_id = {str(c.get("doctor_id")): c for c in for_llm if c.get("doctor_id")}
    for r in recs:
        if str(r.get("reason") or "").strip():
            continue
        c = by_id.get(r["doctor_id"])
        spec = str(c.get("specialties_label") or "").strip() if c else ""
        r["reason"] = (
            "Gợi ý theo độ liên quan kỹ thuật với mô tả của bạn"
            + (f" — chuyên khoa: {spec}" if spec else "")
            + " (không thay tư vấn y khoa)."
        )


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
        spec = str(c.get("specialties_label") or "").strip()
        existing.append(
            {
                "doctor_id": did,
                "reason": (
                    "Gợi ý theo thứ tự tìm kiếm & ranking so với mô tả của bạn"
                    + (f" — {spec}" if spec else "")
                    + " (không thay tư vấn y khoa)."
                ),
            }
        )
    return existing
