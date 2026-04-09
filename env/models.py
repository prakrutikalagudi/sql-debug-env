"""
Typed Pydantic models for SQLDebugEnv.
Observation, Action, Reward — OpenEnv spec compliant.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SQLObservation(BaseModel):
    """What the agent sees at each step."""

    task_id: str = Field(..., description="Unique task identifier")
    task_description: str = Field(..., description="Natural language task description")
    schema_ddl: str = Field(..., description="Full DDL of the database schema")
    sample_data: str = Field(..., description="Sample rows to understand the data")
    current_query: str = Field(..., description="The current (possibly broken) SQL query")
    error_message: Optional[str] = Field(None, description="SQL error from last execution, if any")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    result_preview: Optional[str] = Field(None, description="First 5 rows of query result as text")
    result_row_count: Optional[int] = Field(None, description="Total number of rows returned")
    step_number: int = Field(..., description="Current step (1-indexed)")
    max_steps: int = Field(..., description="Maximum steps allowed")
    hint: Optional[str] = Field(None, description="Optional hint for the agent (task-level)")


class SQLAction(BaseModel):
    """What the agent sends each step."""

    sql_query: str = Field(..., description="The SQL query to execute (agent's attempt)")
    explanation: str = Field(
        default="",
        description="Agent's reasoning about what was wrong and what was changed",
    )


class SQLReward(BaseModel):
    """Shaped reward breakdown for a single step."""

    total: float = Field(..., gt=0.0, lt=1.0, description="Total reward strictly in (0,1)")
    syntax_valid: float = Field(0.0)
    executes: float = Field(0.0)
    result_correct: float = Field(0.0)
    efficient: float = Field(0.0)
    loop_penalty: float = Field(0.0)
    destructive_penalty: float = Field(0.0)


class StepResult(BaseModel):
    """Returned by env.step()."""

    observation: SQLObservation
    reward: float = Field(..., gt=0.0, lt=1.0)   # 🔥 IMPORTANT FIX
    reward_breakdown: SQLReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Returned by env.reset()."""

    observation: SQLObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Returned by env.state() — full internal state snapshot."""

    task_id: str
    step_number: int
    max_steps: int
    cumulative_reward: float
    done: bool
    previous_queries: List[str]
    best_reward_so_far: float
