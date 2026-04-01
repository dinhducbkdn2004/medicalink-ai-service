"""Ghi log JSONL phục vụ đánh giá RAG (query → retrieval → output)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def append_rag_eval(
    log_path: str | None,
    record: dict[str, Any],
) -> None:
    if not log_path or not str(log_path).strip():
        return
    path = str(log_path).strip()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            pass
    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        logger.warning("Không ghi được RAG eval log (%s): %s", path, e)


def build_eval_record(
    *,
    query: str,
    top_k: int,
    retrieved_ids: list[str],
    recommended_ids: list[str],
    hybrid_used: bool,
    legacy_collection: bool,
    rerank_mode: str,
    latency_ms: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query[:4000],
        "top_k": top_k,
        "retrieved_doctor_ids": retrieved_ids,
        "recommended_doctor_ids": recommended_ids,
        "hybrid_used": hybrid_used,
        "legacy_collection": legacy_collection,
        "rerank_mode": rerank_mode,
        "latency_ms": round(latency_ms, 2),
    }
    if extra:
        rec["extra"] = extra
    return rec


class StepTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
