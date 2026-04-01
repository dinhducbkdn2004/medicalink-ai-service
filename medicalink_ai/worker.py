from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aio_pika
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from medicalink_ai.config import Settings, get_settings
from medicalink_ai.rag import DoctorRagService
from medicalink_ai.vector_store import DoctorVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

AI_PATTERN = "ai.doctor-recommendation.request"
DOCTOR_CREATED = "doctor.profile.created"
DOCTOR_UPDATED = "doctor.profile.updated"
DOCTOR_DELETED = "doctor.profile.deleted"


def _nest_rpc_reply(
    channel: aio_pika.abc.AbstractChannel,
    reply_to: str | None,
    correlation_id: str | None,
    body_obj: dict[str, Any],
) -> Any:
    if not reply_to or not correlation_id:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut
    payload = json.dumps(body_obj).encode("utf-8")
    msg = aio_pika.Message(
        body=payload,
        correlation_id=correlation_id,
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )
    return channel.default_exchange.publish(msg, routing_key=reply_to)


async def run_worker(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    if not s.openai_api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY (cần cho embedding).")
    prov = (s.llm_provider or "openai").strip().lower()
    if prov == "gemini" and not (s.google_genai_api_key or "").strip():
        raise RuntimeError(
            "Thiếu GOOGLE_GENAI_API_KEY hoặc GEMINI_API_KEY khi LLM_PROVIDER=gemini."
        )

    openai_client = AsyncOpenAI(api_key=s.openai_api_key)
    qdrant_kw: dict[str, Any] = {"url": s.qdrant_url}
    if (s.qdrant_api_key or "").strip():
        qdrant_kw["api_key"] = s.qdrant_api_key.strip()
    qdrant = AsyncQdrantClient(**qdrant_kw)
    store = DoctorVectorStore(
        qdrant=qdrant,
        openai=openai_client,
        collection_name=s.qdrant_collection_name,
        embedding_model=s.openai_embedding_model,
        openai_api_key=s.openai_api_key,
        hybrid_enabled=s.rag_hybrid_enabled,
        dense_name=s.dense_vector_name,
        sparse_name=s.sparse_vector_name,
        sparse_model_name=s.fastembed_sparse_model,
        prefetch_limit=s.retrieval_prefetch_limit,
    )
    rag = DoctorRagService(
        store=store,
        openai=openai_client,
        settings=s,
    )

    connection = await aio_pika.connect_robust(s.rabbitmq_url)

    # --- RPC ---
    ch_rpc = await connection.channel()
    await ch_rpc.set_qos(prefetch_count=1)
    rpc_queue = await ch_rpc.declare_queue(s.ai_rpc_queue, durable=True)

    # --- Events ---
    ch_ev = await connection.channel()
    await ch_ev.set_qos(prefetch_count=1)
    topic = await ch_ev.declare_exchange(
        s.topic_exchange,
        aio_pika.ExchangeType.TOPIC,
        durable=True,
    )
    ev_queue = await ch_ev.declare_queue(s.ai_events_queue, durable=True)
    for key in (DOCTOR_CREATED, DOCTOR_UPDATED, DOCTOR_DELETED):
        await ev_queue.bind(topic, routing_key=key)

    await store.ensure_collection()
    logger.info(
        "Medicalink AI worker: RPC queue=%s events=%s",
        s.ai_rpc_queue,
        s.ai_events_queue,
    )

    async def handle_rpc(message: aio_pika.IncomingMessage) -> None:
        async with message.process():
            try:
                reply_to = message.reply_to
                if isinstance(reply_to, bytes):
                    reply_to = reply_to.decode("utf-8")
                corr = message.correlation_id
                if isinstance(corr, bytes):
                    corr = corr.decode("utf-8")

                raw = json.loads(message.body.decode("utf-8"))
                pattern = raw.get("pattern")
                data = raw.get("data") or {}
                if pattern != AI_PATTERN:
                    logger.warning("unknown RPC pattern %s", pattern)
                    await _nest_rpc_reply(
                        ch_rpc,
                        reply_to,
                        corr,
                        {
                            "err": {
                                "status": "error",
                                "message": f"Unknown pattern: {pattern}",
                            },
                            "isDisposed": True,
                        },
                    )
                    return
                symptoms = str(data.get("symptoms") or "").strip()
                top_k = int(data.get("topK") or data.get("top_k") or 5)
                top_k = max(1, min(top_k, 15))
                if not symptoms:
                    await _nest_rpc_reply(
                        ch_rpc,
                        reply_to,
                        corr,
                        {
                            "err": {
                                "status": "error",
                                "message": "symptoms is required",
                            },
                            "isDisposed": True,
                        },
                    )
                    return
                result = await rag.recommend(symptoms, top_k)
                await _nest_rpc_reply(
                    ch_rpc,
                    reply_to,
                    corr,
                    {"response": result, "isDisposed": True},
                )
            except Exception as e:
                logger.exception("RPC handler error")
                reply_to_e = message.reply_to
                if isinstance(reply_to_e, bytes):
                    reply_to_e = reply_to_e.decode("utf-8")
                corr_e = message.correlation_id
                if isinstance(corr_e, bytes):
                    corr_e = corr_e.decode("utf-8")
                await _nest_rpc_reply(
                    ch_rpc,
                    reply_to_e,
                    corr_e,
                    {
                        "err": {
                            "status": "error",
                            "message": str(e),
                        },
                        "isDisposed": True,
                    },
                )

    async def handle_event(message: aio_pika.IncomingMessage) -> None:
        async with message.process():
            try:
                raw = json.loads(message.body.decode("utf-8"))
                pattern = str(raw.get("pattern") or "")
                outer = raw.get("data")
                payload: Any = None
                if (
                    isinstance(outer, dict)
                    and "data" in outer
                    and "timestamp" in outer
                ):
                    payload = outer.get("data")
                else:
                    payload = outer

                if pattern == DOCTOR_DELETED:
                    if isinstance(payload, dict) and payload.get("id"):
                        await store.delete_doctor(str(payload["id"]))
                    return

                if pattern in (DOCTOR_CREATED, DOCTOR_UPDATED) and isinstance(
                    payload,
                    dict,
                ):
                    await store.upsert_doctor(payload)
            except Exception:
                logger.exception("event handler error")

    await rpc_queue.consume(handle_rpc)
    await ev_queue.consume(handle_event)

    await asyncio.Future()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
