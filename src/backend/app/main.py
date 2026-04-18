from __future__ import annotations

import asyncio
import io
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import app.sqlite_patch  # noqa: F401 — Chroma needs modern sqlite3

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.graph.workflow import build_debate_graph
from app.config import settings
from app.schemas import (
    AutomlDebateResponse,
    ChatRequest,
    ChatResponse,
    CsvColumnsResponse,
    DebateRunResult,
    DebateRunStatus,
    JudgeDecision,
    ReasoningLogEntry,
)
from app.services.agent_trace import graph_state_to_agent_trace
from app.services.chat_assistant import answer_chat
from app.services.dataset_memory import index_completed_run_from_state
from app.services.run_store import graph_result_to_api, run_store
from app.state import DebateGraphState

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _execute_pipeline(run_id: str, csv_path: str, target_column: str) -> None:
    await run_store.update(run_id, status="running", message="Executing LangGraph pipeline")
    graph = build_debate_graph()
    init: DebateGraphState = _graph_state_init(run_id, csv_path, target_column)
    try:
        final = await asyncio.to_thread(graph.invoke, init)
        if not final.get("error"):
            await asyncio.to_thread(index_completed_run_from_state, run_id, final)
        result = graph_result_to_api(run_id, final)
        await run_store.update(run_id, status="completed" if not result.error else "failed", result=result, message=None)
    except Exception as e:
        result = DebateRunResult(
            run_id=run_id,
            status="failed",
            target_column=target_column,
            task_type="unknown",
            eda_summary="",
            eda_structured=None,
            evaluation_report=None,
            model_runs=[],
            metrics_comparison=[],
            debate_transcript="",
            debate_analysis=None,
            judge=JudgeDecision(winner="none", reason="Pipeline error", confidence=0.0, winner_agent="none"),
            reasoning_logs=[],
            error=str(e),
        )
        await run_store.update(run_id, status="failed", result=result, message=str(e))


@app.post("/api/v1/debate", response_model=DebateRunStatus)
async def start_debate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> DebateRunStatus:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    await asyncio.to_thread(settings.uploads_dir.mkdir, parents=True, exist_ok=True)
    run_id = await run_store.create("", target_column)
    safe_name = Path(file.filename).name
    dest = settings.uploads_dir / f"{run_id}_{safe_name}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    await run_store.update(run_id, csv_path=str(dest))
    background_tasks.add_task(_execute_pipeline, run_id, str(dest), target_column)
    data = await run_store.get(run_id)
    st = run_store.to_status(run_id, data)
    assert st is not None
    return st


@app.get("/api/v1/debate/{run_id}", response_model=DebateRunStatus)
async def get_debate(run_id: str) -> DebateRunStatus:
    data = await run_store.get(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    st = run_store.to_status(run_id, data)
    assert st is not None
    return st


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    ML advisor chat: answers general algorithm/dataset questions and questions about the active run
    (using the optional ``run_context`` snapshot).
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages[] must include at least one user message.")
    try:
        raw = await asyncio.to_thread(
            answer_chat,
            [m.model_dump() for m in req.messages],
            req.run_context,
        )
    except Exception as e:
        logger.exception("chat assistant failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e!s}") from e
    return ChatResponse(reply=str(raw.get("reply", "")), model=str(raw.get("model", "")))


_MAX_CSV_HEADER_SCAN_BYTES = 2 * 1024 * 1024


@app.post("/api/v1/dataset/csv-columns", response_model=CsvColumnsResponse)
async def csv_columns(file: UploadFile = File(..., description="CSV file; only the header is parsed.")) -> CsvColumnsResponse:
    """
    Read the first row of a CSV upload and return column names for target-column pickers.
    Consumes at most the first ~2 MiB of the file (sufficient for wide headers).
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Expected a file with .csv extension.")

    try:
        chunk = await file.read(_MAX_CSV_HEADER_SCAN_BYTES)
    except Exception as e:
        logger.exception("csv-columns read failed")
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e!s}") from e

    if not chunk:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(chunk), nrows=0)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV header. Ensure the file is comma-separated with a header row: {e!s}",
        ) from e

    columns = [str(c) for c in df.columns.tolist()]
    if not columns:
        raise HTTPException(status_code=400, detail="No columns found in CSV header.")

    return CsvColumnsResponse(columns=columns)


def _graph_state_init(run_id: str, csv_path: str, target_column: str) -> DebateGraphState:
    return {
        "run_id": run_id,
        "csv_path": csv_path,
        "target_column": target_column,
        "dataset_bundle": {},
        "eda_structured": {},
        "model_proposals": {},
        "model_runs": {},
        "metrics": {},
        "evaluation_report": {},
        "reasoning_logs": [],
        "memory_context": "",
        "memory_hits": [],
    }


def _invoke_automl_pipeline_sync(run_id: str, csv_path: str, target_column: str) -> dict[str, Any]:
    graph = build_debate_graph()
    init: DebateGraphState = _graph_state_init(run_id, csv_path, target_column)
    return graph.invoke(init)


def _final_state_to_automl_response(final: dict[str, Any]) -> AutomlDebateResponse:
    eda = final.get("eda_structured")
    if eda is None or not isinstance(eda, dict):
        eda = {}
    metrics_raw = final.get("metrics") or {}
    metrics: dict[str, Any] = {}
    if isinstance(metrics_raw, dict):
        for k, v in metrics_raw.items():
            metrics[str(k)] = dict(v) if isinstance(v, dict) else v

    models: list[dict[str, Any]] = []
    runs = final.get("model_runs") or {}
    if isinstance(runs, dict):
        for key in sorted(runs.keys()):
            info = runs[key]
            if not isinstance(info, dict):
                continue
            row: dict[str, Any] = {
                "model_key": key,
                "agent": info.get("agent", ""),
                "model_type": info.get("model_type", ""),
                "artifact_path": info.get("artifact_path", ""),
                "params": info.get("params") or {},
                "metrics": metrics.get(key),
                "proposal": info.get("proposal"),
                "evaluation": info.get("evaluation"),
            }
            models.append(row)

    debate = str(final.get("debate_transcript") or "")
    judge = final.get("judge_decision") or {}
    winner = str(judge.get("winner") or "none")
    winner_agent = str(judge.get("winner_agent") or "")
    judge_reason = str(judge.get("reason") or "")
    try:
        judge_confidence = float(judge.get("confidence", 0.0))
    except (TypeError, ValueError):
        judge_confidence = 0.0

    reasoning_logs: list[Any] = []
    raw_logs = final.get("reasoning_logs") or []
    if isinstance(raw_logs, list):
        for item in raw_logs:
            if not isinstance(item, dict):
                continue
            try:
                reasoning_logs.append(
                    ReasoningLogEntry(
                        agent=str(item.get("agent", "")),
                        step=str(item.get("step", "")),
                        content=str(item.get("content", "")),
                        metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    )
                )
            except Exception:
                continue

    agent_trace = graph_state_to_agent_trace(final)

    return AutomlDebateResponse(
        eda=eda,
        models=models,
        debate=debate,
        winner=winner,
        winner_agent=winner_agent,
        judge_reason=judge_reason,
        judge_confidence=judge_confidence,
        metrics=metrics,
        reasoning_logs=reasoning_logs,
        agent_trace=agent_trace,
    )


@app.post("/automl-debate", response_model=AutomlDebateResponse)
async def automl_debate(
    file: UploadFile = File(..., description="Tabular dataset as CSV."),
    target_column: str = Form(..., description="Name of the target / label column."),
) -> AutomlDebateResponse:
    """
    Run the full AutoML debate pipeline synchronously (LangGraph in a worker thread) and return structured results.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Expected a file with .csv extension.")

    tc = (target_column or "").strip()
    if not tc:
        raise HTTPException(status_code=400, detail="target_column must be a non-empty string.")

    await asyncio.to_thread(settings.uploads_dir.mkdir, parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    dest = settings.uploads_dir / f"automl_{run_id}_{safe_name}"

    try:
        body = await file.read()
        if not body:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        await asyncio.to_thread(dest.write_bytes, body)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to store upload for automl-debate")
        raise HTTPException(status_code=500, detail=f"Could not store uploaded file: {e!s}") from e

    try:
        final = await asyncio.to_thread(_invoke_automl_pipeline_sync, run_id, str(dest), tc)
    except Exception as e:
        logger.exception("automl-debate pipeline crashed")
        try:
            await asyncio.to_thread(dest.unlink, missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e!s}") from e

    err = final.get("error")
    if err:
        try:
            await asyncio.to_thread(dest.unlink, missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=str(err))

    try:
        await asyncio.to_thread(index_completed_run_from_state, run_id, final)
    except Exception as e:
        logger.warning("Dataset memory indexing failed (non-fatal): %s", e)

    return _final_state_to_automl_response(final)
