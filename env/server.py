import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import SQLDebugEnv
from env.models import ResetResult, SQLAction, StateResult
from env.tasks import TASKS

_envs: Dict[str, SQLDebugEnv] = {}
DEFAULT_TASK = os.getenv("SQL_DEBUG_TASK", "syntax_fix")

EPS = 0.001



def clamp(v: float) -> float:
    return max(0.001, min(0.999, float(v)))


def _get_or_create_env(task_id: str) -> SQLDebugEnv:
    if task_id not in _envs:
        _envs[task_id] = SQLDebugEnv(task_id=task_id)
    return _envs[task_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for task_id in TASKS:
        env = _get_or_create_env(task_id)
        env.reset()
    yield


app = FastAPI(title="SQLDebugEnv", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    task_id: str = DEFAULT_TASK


class StepRequest(BaseModel):
    task_id: str = DEFAULT_TASK
    sql_query: str
    explanation: str = ""


class StepResponse(BaseModel):
    """
    Clean step response — reward_breakdown component fields (which can be 0.0)
    are excluded from the top-level response to avoid validator false positives.
    Only `reward` (the clamped total) is exposed as a scored float.
    """
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]


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
    if request is None:
        request = ResetRequest()
    task_id = request.task_id if request and request.task_id else DEFAULT_TASK
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    env = _get_or_create_env(task_id)
    return env.reset()


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    if request.task_id not in TASKS:
        raise HTTPException(
            status_code=400, detail=f"Unknown task_id '{request.task_id}'"
        )
    env = _get_or_create_env(request.task_id)
    if env._conn is None:
        env.reset()
    action = SQLAction(sql_query=request.sql_query, explanation=request.explanation)
    try:
        result = env.step(action)
    except RuntimeError:
        env.reset()
        result = env.step(action)

    # Clamp reward strictly inside (0, 1) — never 0.0 or 1.0
    clamped_reward = clamp(result.reward)

    # Build reward_breakdown as plain info (not scored fields)
    breakdown = result.reward_breakdown.model_dump()

    return StepResponse(
        observation=result.observation,
        reward=clamped_reward,
        done=result.done,
        info={
            "solved": result.info.get("solved", False),
            "steps_remaining": result.info.get("steps_remaining", 0),
        },
    )


@app.get("/state", response_model=StateResult)
async def state(task_id: str = DEFAULT_TASK):
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400, detail=f"Unknown task_id '{task_id}'"
        )
    env = _get_or_create_env(task_id)
    return env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
