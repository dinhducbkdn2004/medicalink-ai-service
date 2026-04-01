"""Re-ranking nhẹ sau retrieval: FlashRank (cross-encoder nhẹ) hoặc tăng điểm lexical."""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Any, Literal

logger = logging.getLogger(__name__)

RerankMode = Literal["none", "lexical", "flashrank"]


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t


def _tokens(text: str) -> set[str]:
    t = _normalize(text)
    return {x for x in re.split(r"[^\w]+", t) if len(x) >= 2}


def lexical_bonus(query: str, candidate: dict[str, Any]) -> float:
    """Điểm 0..1: overlap token giữa query và tên + chuyên khoa + snippet JSON."""
    qset = _tokens(query)
    if not qset:
        return 0.0
    blob = " ".join(
        str(candidate.get(k) or "")
        for k in ("full_name", "specialties_label", "source_json")
    )
    cset = _tokens(blob)
    if not cset:
        return 0.0
    inter = len(qset & cset)
    return min(1.0, inter / max(3.0, len(qset) * 0.5))


def blend_scores(
    query: str,
    candidates: list[dict[str, Any]],
    lexical_weight: float,
) -> list[dict[str, Any]]:
    if not candidates or lexical_weight <= 0:
        return candidates
    scores = [float(c.get("score") or 0.0) for c in candidates]
    mx = max(scores) if scores else 1.0
    mn = min(scores) if scores else 0.0
    span = mx - mn or 1.0
    enriched: list[tuple[float, dict[str, Any]]] = []
    for c in candidates:
        base = float(c.get("score") or 0.0)
        norm = (base - mn) / span
        lex = lexical_bonus(query, c)
        final = norm * (1.0 - lexical_weight) + lex * lexical_weight
        nc = dict(c)
        nc["score"] = final
        nc["rerank_lexical"] = lex
        enriched.append((final, nc))
    enriched.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in enriched]


@lru_cache(maxsize=2)
def _flashrank_ranker(model_name: str, cache_dir: str):
    from flashrank import Ranker

    return Ranker(model_name=model_name, cache_dir=cache_dir)


def rerank_flashrank(
    query: str,
    candidates: list[dict[str, Any]],
    model_name: str = "ms-marco-MiniLM-L-12-v2",
    cache_dir: str | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    from flashrank import RerankRequest

    import tempfile

    cdir = cache_dir or tempfile.mkdtemp(prefix="medicalink_flashrank_")
    try:
        ranker = _flashrank_ranker(model_name, cdir)
    except Exception as e:
        logger.warning("FlashRank không khởi tạo được, bỏ qua: %s", e)
        return candidates
    passages: list[dict[str, str]] = []
    for i, c in enumerate(candidates):
        did = str(c.get("doctor_id") or i)
        text_parts = [
            str(c.get("full_name") or ""),
            str(c.get("specialties_label") or ""),
            (str(c.get("source_json") or ""))[:1200],
        ]
        passages.append({"id": did, "text": " | ".join(p for p in text_parts if p)})
    try:
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    except Exception as e:
        logger.warning("FlashRank.rerank lỗi: %s", e)
        return candidates
    by_id = {str(c.get("doctor_id")): dict(c) for c in candidates if c.get("doctor_id")}
    out: list[dict[str, Any]] = []
    for p in ranked:
        rid = str(p.get("id") or "")
        if rid in by_id:
            row = dict(by_id[rid])
            if "score" in p:
                row["score"] = float(p["score"])
            row["rerank_flashrank"] = True
            out.append(row)
    for c in candidates:
        cid = str(c.get("doctor_id") or "")
        if cid and cid not in {str(x.get("doctor_id")) for x in out}:
            out.append(dict(c))
    if top_n is not None:
        out = out[:top_n]
    return out


def rerank_pipeline(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    mode: RerankMode,
    lexical_weight: float,
    flashrank_model: str,
    flashrank_cache_dir: str | None,
    flashrank_pool: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pool = candidates[: min(len(candidates), flashrank_pool)]
    if mode == "none":
        return pool
    if mode == "lexical":
        return blend_scores(query, pool, lexical_weight=max(0.0, min(lexical_weight, 0.9)))
    if mode == "flashrank":
        fr = rerank_flashrank(
            query,
            pool,
            model_name=flashrank_model,
            cache_dir=flashrank_cache_dir,
        )
        if lexical_weight > 0:
            return blend_scores(query, fr, lexical_weight=lexical_weight * 0.5)
        return fr
    return pool
