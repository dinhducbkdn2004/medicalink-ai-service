from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rabbitmq_url: str = "amqp://admin:admin123@localhost:5672/"
    # Embedding dense: OpenAI (LangChain). LLM có thể Gemini (xem llm_provider).
    openai_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "doctor_profiles"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    # openai | gemini — khớp biến microservice GOOGLE_GENAI_*
    llm_provider: str = "openai"
    google_genai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_GENAI_API_KEY", "GEMINI_API_KEY"),
    )
    google_genai_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias=AliasChoices(
            "GOOGLE_GENAI_MODEL",
            "GEMINI_MODEL",
        ),
    )
    google_genai_timeout_ms: int = Field(
        default=120_000,
        validation_alias=AliasChoices("GOOGLE_GENAI_TIMEOUT", "GEMINI_TIMEOUT_MS"),
    )
    rag_llm_temperature: float = 0.2
    api_gateway_base_url: str = "http://localhost:3000"

    ai_rpc_queue: str = "ai_service_queue"
    ai_events_queue: str = "ai_service_events_queue"
    topic_exchange: str = "medicalink.topic"

    # --- Hybrid Qdrant (dense OpenAI + sparse FastEmbed / BM25) ---
    rag_hybrid_enabled: bool = True
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "lexical"
    fastembed_sparse_model: str = "Qdrant/bm25"
    # Prefetch / hybrid pool; RRF rồi rerank; LLM chỉ nhận retrieval_llm_context_max bản ghi.
    retrieval_prefetch_limit: int = 40
    retrieval_llm_context_max: int = 24

    # Re-ranking: none | lexical | flashrank
    rag_rerank_mode: str = "flashrank"
    rag_rerank_lexical_weight: float = 0.2
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = ".cache/flashrank"
    rag_rerank_pool: int = 36

    # Để trống = tắt. Ví dụ: ./data/rag_eval.jsonl
    rag_eval_log_path: str = ""


def get_settings() -> Settings:
    return Settings()
