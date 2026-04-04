"""
FastAPI server for SQLDebugEnv.
Exposes OpenEnv-compliant REST endpoints:
  POST /reset         → ResetResult
  POST /step          → StepResult
  GET  /state         → StateResult
  GET  /tasks         → list of available tasks
  GET  /health        → health check
"""

import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import SQLDebugEnv
from env.models import ResetResult, SQLAction, StateResult, StepResult
from env.tasks import TASKS


# ──────────────────────────────────────────────
# Global env store (keyed by session_id)
# For simplicity, single default env + per-task named envs
# ──────────────────────────────────────────────

_envs: Dict[str, SQLDebugEnv] = {}

DEFAULT_TASK = os.getenv("SQL_DEBUG_TASK", "syntax_fix")


def _get_or_create_env(task_id: str) -> SQLDebugEnv:
    if task_id not in _envs:
        _envs[task_id] = SQLDebugEnv(task_id=task_id)
    return _envs[task_id]


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the default environment
    env = _get_or_create_env(DEFAULT_TASK)
    env.reset()
    yield


app = FastAPI(
    title="SQLDebugEnv",
    description=(
        "An OpenEnv-compliant environment where AI agents debug broken SQL queries. "
        "Three tasks ranging from syntax errors (easy) to logical bugs (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = DEFAULT_TASK


class StepRequest(BaseModel):
    task_id: str = DEFAULT_TASK
    sql_query: str
    explanation: str = ""


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}


@app.get("/tasks")
async def list_tasks():
    return {
        tid: {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "description": t.description,
            "max_steps": t.max_steps,
        }
        for tid, t in TASKS.items()
    }


@app.post("/reset", response_model=ResetResult)
async def reset(request: ResetRequest = None):
    """Reset the environment for the given task. Returns initial observation."""
    if request is None:
        request = ResetRequest()
    task_id = request.task_id if request and request.task_id else DEFAULT_TASK
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    env = _get_or_create_env(task_id)
    return env.reset()


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest):
    """Execute the agent's SQL query and return reward + next observation."""
    if request.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{request.task_id}'")
    env = _get_or_create_env(request.task_id)
    action = SQLAction(sql_query=request.sql_query, explanation=request.explanation)
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateResult)
async def state(task_id: str = DEFAULT_TASK):
    """Return current internal state snapshot."""
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    env = _get_or_create_env(task_id)
    return env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
