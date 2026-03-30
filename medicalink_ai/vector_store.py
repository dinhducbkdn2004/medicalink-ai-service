from __future__ import annotations

import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from medicalink_ai.doctor_knowledge import build_doctor_document, doctor_point_id

logger = logging.getLogger(__name__)

# text-embedding-3-small
DEFAULT_VECTOR_SIZE = 1536


class DoctorVectorStore:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        openai: AsyncOpenAI,
        collection_name: str,
        embedding_model: str,
        openai_api_key: str,
    ) -> None:
        self.qdrant = qdrant
        self.openai = openai
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key,
        )

    async def ensure_collection(self) -> None:
        try:
            await self.qdrant.get_collection(self.collection_name)
            return
        except Exception:
            pass
        await self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=DEFAULT_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection %s", self.collection_name)

    async def embed_text(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text.replace("\n", " ")[:8000])

    async def upsert_doctor(self, profile: dict[str, Any]) -> None:
        await self.ensure_collection()
        text, payload = build_doctor_document(profile)
        if not payload["doctor_id"]:
            logger.warning("skip upsert: missing doctor id in profile")
            return
        vector = await self.embed_text(text)
        pid = doctor_point_id(payload["doctor_id"])
        await self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=pid,
                    vector=vector,
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

    async def search_active(
        self,
        query_text: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection()
        vector = await self.embed_text(query_text)
        flt = Filter(
            must=[
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]
        )
        hits = await self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        out: list[dict[str, Any]] = []
        for h in hits:
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
