from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rabbitmq_url: str = "amqp://admin:admin123@localhost:5672/"
    openai_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "doctor_profiles"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    api_gateway_base_url: str = "http://localhost:3000"

    ai_rpc_queue: str = "ai_service_queue"
    ai_events_queue: str = "ai_service_events_queue"
    topic_exchange: str = "medicalink.topic"


def get_settings() -> Settings:
    return Settings()
