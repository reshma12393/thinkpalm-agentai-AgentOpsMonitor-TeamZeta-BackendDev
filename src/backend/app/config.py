from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Loads from process environment and ``src/backend/.env`` (when present).

    LLM (optional) — **OpenRouter** (OpenAI-compatible API):
    - ``OPENROUTER_API_KEY`` — required for chat models when using the LLM path
    - ``OPENROUTER_BASE_URL`` — default ``https://openrouter.ai/api/v1``
    - ``OPENROUTER_MODEL`` — OpenRouter model id (e.g. ``anthropic/claude-haiku-4.5``)
    - ``OPENROUTER_HTTP_REFERER`` / ``OPENROUTER_APP_TITLE`` — optional OpenRouter attribution headers
    - ``OPENROUTER_MAX_OUTPUT_TOKENS`` — max completion tokens per call (caps cost; avoids huge default e.g. 64k)
    - ``OPENROUTER_MAX_LLM_INPUT_CHARS`` — truncate JSON/text prompts to this many characters (input savings)
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AutoML Debate System"
    data_dir: Path = BACKEND_DIR / "data"
    uploads_dir: Path = BACKEND_DIR / "data" / "uploads"
    runs_dir: Path = BACKEND_DIR / "data" / "runs"
    chroma_dir: Path = BACKEND_DIR / "data" / "chroma"

    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key (sk-or-v1-...).")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter OpenAI-compatible base URL.",
    )
    openrouter_model: str = Field(
        default="anthropic/claude-haiku-4.5",
        description="Model slug from OpenRouter models page.",
    )
    openrouter_http_referer: str | None = Field(
        default=None,
        description="Optional HTTP-Referer header for OpenRouter rankings.",
    )
    openrouter_app_title: str | None = Field(
        default=None,
        description="Optional X-Title header for OpenRouter attribution.",
    )
    openrouter_max_output_tokens: int = Field(
        default=2048,
        ge=256,
        le=16384,
        description="Max tokens the model may generate per call (lowers reserved credits vs omitting max_tokens).",
    )
    openrouter_max_llm_input_chars: int = Field(
        default=10000,
        ge=2000,
        le=100_000,
        description="Budget for the largest JSON blob in a prompt; other excerpts scale from this (input savings).",
    )

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.runs_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
