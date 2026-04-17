from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(temperature: float = 0.2) -> BaseChatModel | None:
    """
    Chat LLM via **OpenRouter** (OpenAI-compatible ``/v1/chat/completions``).

    Set ``OPENROUTER_API_KEY`` in ``src/backend/.env``. Pick ``OPENROUTER_MODEL`` from
    https://openrouter.ai/models (e.g. ``anthropic/claude-haiku-4.5``).
    """
    key = (settings.openrouter_api_key or "").strip()
    if not key:
        return None
    base = (settings.openrouter_base_url or "").strip().rstrip("/")
    model = (settings.openrouter_model or "").strip()
    if not base or not model:
        return None

    headers: dict[str, str] = {}
    ref = (settings.openrouter_http_referer or "").strip()
    if ref:
        headers["HTTP-Referer"] = ref
    title = (settings.openrouter_app_title or "").strip()
    if title:
        headers["X-Title"] = title

    max_out = int(settings.openrouter_max_output_tokens)
    max_out = max(256, min(max_out, 16384))

    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=base,
        temperature=temperature,
        max_tokens=max_out,
        default_headers=headers or None,
    )
