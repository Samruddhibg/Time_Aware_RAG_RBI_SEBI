"""
core/config.py — Centralised settings loaded from environment / .env file.
All other modules import `settings` from here.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://raguser:ragpass@localhost:5432/ragdb",
        description="asyncpg-compatible connection string",
    )

    # ── Embedder ──────────────────────────────────────────────────────────────
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embed_dim: int = Field(default=384)
    embed_batch: int = Field(default=64)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_model: str = Field(default="llama3.2:3b")
    llm_api_key: str = Field(default="ollama")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=1024)
    llm_timeout: int = Field(default=60)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_dense: int = Field(default=20)
    top_k_lexical: int = Field(default=20)
    top_k_rerank: int = Field(default=6)
    min_sim_threshold: float = Field(default=0.35)

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000")
    log_level: str = Field(default="info")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
