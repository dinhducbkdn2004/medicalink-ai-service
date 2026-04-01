from __future__ import annotations

import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from medicalink_ai.doctor_knowledge import build_doctor_document, doctor_point_id
from medicalink_ai.sparse_encoder import text_to_sparse_vector

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_SIZE = 1536


class DoctorVectorStore:
    """Qdrant: hybrid (dense + sparse) hoặc legacy (chỉ dense, collection cũ)."""

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        openai: AsyncOpenAI,
        collection_name: str,
        embedding_model: str,
        openai_api_key: str,
        *,
        hybrid_enabled: bool = True,
        dense_name: str = "dense",
        sparse_name: str = "lexical",
        sparse_model_name: str = "Qdrant/bm25",
        prefetch_limit: int = 40,
    ) -> None:
        self.qdrant = qdrant
        self.openai = openai
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key,
        )
        self.hybrid_enabled = hybrid_enabled
        self.dense_name = dense_name
        self.sparse_name = sparse_name
        self.sparse_model_name = sparse_model_name
        self.prefetch_limit = prefetch_limit
        self._legacy_single_vector: bool | None = None

    async def embed_text(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text.replace("\n", " ")[:8000])

    async def _read_legacy_flag(self) -> bool:
        if self._legacy_single_vector is not None:
            return self._legacy_single_vector
        try:
            info = await self.qdrant.get_collection(self.collection_name)
        except Exception:
            self._legacy_single_vector = False
            return False
        sparse = info.config.params.sparse_vectors
        has_sparse = bool(sparse and len(sparse) > 0)
        self._legacy_single_vector = not has_sparse
        if self._legacy_single_vector and self.hybrid_enabled:
            logger.warning(
                "Collection %s không có sparse vectors — chỉ dùng dense search. "
                "Để bật hybrid: tạo lại collection hoặc đổi QDRANT_COLLECTION_NAME.",
                self.collection_name,
            )
        return self._legacy_single_vector

    async def ensure_collection(self) -> None:
        try:
            await self.qdrant.get_collection(self.collection_name)
            await self._read_legacy_flag()
            return
        except Exception:
            pass

        if self.hybrid_enabled:
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_name: VectorParams(
                        size=DEFAULT_VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    self.sparse_name: SparseVectorParams(),
                },
            )
            self._legacy_single_vector = False
            logger.info(
                "Created hybrid Qdrant collection %s (%s + %s)",
                self.collection_name,
                self.dense_name,
                self.sparse_name,
            )
        else:
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=DEFAULT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            self._legacy_single_vector = True
            logger.info("Created legacy Qdrant collection %s", self.collection_name)

    async def upsert_doctor(self, profile: dict[str, Any]) -> None:
        await self.ensure_collection()
        text, payload = build_doctor_document(profile)
        if not payload["doctor_id"]:
            logger.warning("skip upsert: missing doctor id in profile")
            return
        dense = await self.embed_text(text)
        pid = doctor_point_id(payload["doctor_id"])
        legacy = await self._read_legacy_flag()
        if legacy:
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=pid,
                        vector=dense,
                        payload=payload,
                    )
                ],
            )
        else:
            sparse = text_to_sparse_vector(text, self.sparse_model_name)
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=pid,
                        vector={
                            self.dense_name: dense,
                            self.sparse_name: sparse,
                        },
                        payload=payload,
                    )
                ],
            )
        logger.info("Upserted doctor vector %s", payload["doctor_id"])

    async def delete_doctor(self, doctor_profile_id: str) -> None:
        await self.ensure_collection()
        pid = doctor_point_id(doctor_profile_id)
        await self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=[pid],
        )
        logger.info("Deleted doctor vector %s", doctor_profile_id)

    def _payload_hits(self, scored: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for h in scored:
            pl = h.payload or {}
            out.append(
                {
                    "doctor_id": pl.get("doctor_id"),
                    "full_name": pl.get("full_name"),
                    "score": h.score,
                    "specialties_label": pl.get("specialties_label"),
                    "source_json": pl.get("source_json"),
                }
            )
        return out

    async def search_active(
        self,
        query_text: str,
        limit: int,
        *,
        filter_specialty_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        """
        Trả về (candidates, hybrid_query_used, legacy_collection).

        - hybrid_query_used: True nếu chạy prefetch dense+sparse + RRF.
        - legacy_collection: True nếu collection chỉ có dense (schema cũ).
        - filter_specialty_ids: optional MatchAny on payload specialty_ids.
        """
        await self.ensure_collection()
        legacy = await self._read_legacy_flag()
        must = [
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
        if filter_specialty_ids:
            ids = [str(x).strip() for x in filter_specialty_ids if str(x).strip()]
            if ids:
                must.append(
                    FieldCondition(key="specialty_ids", match=MatchAny(any=ids)),
                )
        flt = Filter(must=must)
        vector = await self.embed_text(query_text)

        if legacy or not self.hybrid_enabled:
            hits = await self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
            return self._payload_hits(hits), False, legacy

        sparse = text_to_sparse_vector(query_text, self.sparse_model_name)
        prefetch_limit = max(limit, self.prefetch_limit)
        res = await self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                qm.Prefetch(
                    query=vector,
                    using=self.dense_name,
                    filter=flt,
                    limit=prefetch_limit,
                ),
                qm.Prefetch(
                    query=sparse,
                    using=self.sparse_name,
                    filter=flt,
                    limit=prefetch_limit,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        pts = res.points or []
        return self._payload_hits(pts), True, False
