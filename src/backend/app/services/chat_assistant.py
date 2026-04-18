"""ML advisor chat: general dataset→algorithm guidance plus Q&A about the active AutoML run."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.llm_util import get_chat_model
from app.config import settings


SYSTEM_PROMPT = """\
You are the AutoML Arena advisor, an ML assistant embedded in a multi-agent model-selection app.

You answer TWO kinds of questions:

1. General ML guidance — which algorithm fits which dataset shape (small/large, tabular,
   high-dimensional, imbalanced, linear vs non-linear); trade-offs between RandomForest, XGBoost,
   Logistic Regression, Ridge; evaluation, imbalance, overfitting basics.

2. Questions about the CURRENT RUN when a ``run_context`` block is provided — EDA, metrics,
   overfitting signals, debate transcript, judge reasoning, why one model won.

HARD RESPONSE RULES (follow strictly):
- ALWAYS respond in **4 sentences or fewer**. Treat each sentence as a complete thought ending with a period.
- Summary style: give the conclusion first, then at most one supporting sentence with key numbers or caveats.
- When ``run_context`` exists, cite its actual numbers (e.g. ``f1_macro=0.84``, ``imbalance_ratio=2.1``). Never invent metrics or columns.
- If the question needs run_context that is missing, say so briefly in one sentence, then answer the general version.
- No bullet lists, no headings, no chain-of-thought — just the concise summary sentences.
"""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


# Splits on sentence enders followed by whitespace/EOL; keeps the punctuation with each sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def _limit_sentences(text: str, max_sentences: int = 4) -> str:
    """Keep at most ``max_sentences`` sentences of the reply; strips bullets/headings to match summary style."""
    if not text:
        return text
    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+\.\s+|#{1,6}\s+|>\s+)", "", line)
        cleaned_lines.append(line)
    flat = " ".join(cleaned_lines).strip()
    if not flat:
        return flat
    sentences = _SENTENCE_RE.split(flat)
    kept = [s.strip() for s in sentences if s.strip()][:max_sentences]
    out = " ".join(kept)
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _summarize_run_context(run_context: dict[str, Any] | None) -> str | None:
    """Trim the browser-side run payload to a compact JSON block fit for an LLM prompt."""
    if not isinstance(run_context, dict) or not run_context:
        return None

    budget = int(settings.openrouter_max_llm_input_chars)

    eda = run_context.get("eda") if isinstance(run_context.get("eda"), dict) else {}
    det = eda.get("deterministic") if isinstance(eda.get("deterministic"), dict) else {}
    summary = {
        "task_type": run_context.get("task_type"),
        "target_column": run_context.get("target_column"),
        "eda": {
            "n_rows": det.get("n_rows"),
            "n_features": det.get("n_features"),
            "target_profile": det.get("target_profile"),
            "class_imbalance": det.get("class_imbalance"),
            "missing_values": {
                "fraction_rows_any_missing": (det.get("missing_values") or {}).get("fraction_rows_any_missing"),
                "top_columns": list((det.get("missing_values") or {}).get("by_column", {}).items())[:10],
            },
            "correlations": {
                "top_feature_pairs": (det.get("correlations") or {}).get("top_feature_pairs", [])[:8],
                "target_vs_numeric": (det.get("correlations") or {}).get("target_vs_numeric", [])[:8],
            },
            "feature_types": det.get("feature_types"),
        },
        "models": run_context.get("models") or [],
        "metrics": run_context.get("metrics") or {},
        "winner": run_context.get("winner"),
        "winner_agent": run_context.get("winner_agent"),
        "judge_reason": run_context.get("judge_reason"),
        "judge_confidence": run_context.get("judge_confidence"),
        "debate_excerpt": _clip(str(run_context.get("debate") or ""), max(1000, budget // 2)),
    }

    text = json.dumps(summary, default=str, ensure_ascii=False)
    return _clip(text, budget)


def _coerce_role(role: str | None) -> str:
    r = (role or "user").lower()
    if r not in {"user", "assistant", "system"}:
        return "user"
    return r


def answer_chat(messages: list[dict[str, Any]], run_context: dict[str, Any] | None) -> dict[str, Any]:
    """Run a single chat turn through the configured LLM; returns ``{"reply": str, "model": str}``.

    Falls back to a helpful message when the LLM is not configured.
    """
    llm = get_chat_model(temperature=0.2)
    if llm is None:
        return {
            "reply": (
                "The chat advisor isn't configured on this backend — set `OPENROUTER_API_KEY` in `src/backend/.env`. "
                "Meanwhile: open the **Agent trace** and **Model comparison** sections to inspect the run."
            ),
            "model": "(disabled)",
        }

    lc_messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    ctx_text = _summarize_run_context(run_context)
    if ctx_text:
        lc_messages.append(
            SystemMessage(
                content=(
                    "Active run context (trusted JSON; cite numbers from here when answering run-specific questions):\n"
                    f"{ctx_text}"
                )
            )
        )

    history = messages[-12:]
    for m in history:
        role = _coerce_role(m.get("role"))
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))

    if len(lc_messages) <= 1 or not isinstance(lc_messages[-1], HumanMessage):
        return {"reply": "Please send a question to get started.", "model": settings.openrouter_model}

    resp = llm.invoke(lc_messages)
    reply = getattr(resp, "content", None)
    if isinstance(reply, list):
        reply = " ".join(str(p.get("text") if isinstance(p, dict) else p) for p in reply)
    reply = str(reply or "").strip()
    if not reply:
        reply = "I didn't receive any content from the model. Try rephrasing or asking a narrower question."
    reply = _limit_sentences(reply, max_sentences=4)

    return {"reply": reply, "model": settings.openrouter_model}
