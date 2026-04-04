"""
SQLDebugEnv — core environment implementation.

Simulates a SQL debugging workbench: the agent receives a broken query,
executes fixed versions against an in-process SQLite database, and receives
shaped rewards for syntax validity, execution success, correctness, and efficiency.
"""

import sqlite3
from typing import Any, Dict, List, Optional

from env.graders import GRADER_MAP, execute_query
from env.models import (
    ResetResult,
    SQLAction,
    SQLObservation,
    StateResult,
    StepResult,
)
from env.tasks import TASKS, Task


def _build_db(task: Task) -> sqlite3.Connection:
    """Create an in-memory SQLite DB, apply schema and seed data."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(task.schema_ddl)
    for stmt in task.seed_sql:
        conn.execute(stmt)
    conn.commit()
    return conn


def _format_preview(rows: Optional[List[Any]], limit: int = 5) -> Optional[str]:
    if not rows:
        return "(no rows returned)"
    lines = [str(row) for row in rows[:limit]]
    suffix = f"\n... ({len(rows)} rows total)" if len(rows) > limit else f"\n({len(rows)} rows total)"
    return "\n".join(lines) + suffix


def _get_sample_data(conn: sqlite3.Connection, schema_ddl: str) -> str:
    """Pull 3 sample rows from each table for the observation."""
    tables = []
    for line in schema_ddl.splitlines():
        line = line.strip()
        if line.upper().startswith("CREATE TABLE"):
            tname = line.split()[2].strip("(")
            tables.append(tname)

    parts = []
    for tname in tables:
        try:
            cur = conn.execute(f"SELECT * FROM {tname} LIMIT 3")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            parts.append(f"-- {tname}: {cols}")
            for r in rows:
                parts.append(f"   {r}")
        except Exception:
            pass
    return "\n".join(parts)


class SQLDebugEnv:
    """
    OpenEnv-compliant SQL debugging environment.

    Lifecycle:
        env = SQLDebugEnv(task_id="syntax_fix")
        reset_result = env.reset()
        step_result  = env.step(SQLAction(sql_query="SELECT ..."))
        state        = env.state()
    """

    def __init__(self, task_id: str = "syntax_fix"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")
        self.task_id = task_id
        self._task: Task = TASKS[task_id]
        self._conn: Optional[sqlite3.Connection] = None
        self._gold_rows: Optional[List[Any]] = None
        self._step: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._best_reward: float = 0.0
        self._prev_queries: List[str] = []

    # ──────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────

    def reset(self) -> ResetResult:
        """Reset the environment and return the initial observation."""
        self._conn = _build_db(self._task)
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._best_reward = 0.0
        self._prev_queries = []

        # Compute gold results once (deterministic per episode)
        _, self._gold_rows, _, _ = execute_query(self._conn, self._task.gold_query)

        obs = self._make_observation(
            current_query=self._task.broken_query,
            error_message=None,
            execution_time_ms=None,
            result_preview=None,
            result_row_count=None,
        )
        return ResetResult(observation=obs, info={"task_difficulty": self._task.difficulty})

    def step(self, action: SQLAction) -> StepResult:
        """Execute agent's SQL query and return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        query = action.sql_query.strip()

        # Run grader
        grader_fn = GRADER_MAP[self.task_id]
        reward_obj = grader_fn(
            conn=self._conn,
            agent_query=query,
            gold_rows=self._gold_rows,
            prev_queries=self._prev_queries,
        )

        # Execute for observation (re-run to capture preview)
        success, rows, error_msg, elapsed_ms = execute_query(self._conn, query)
        preview = _format_preview(rows) if success else None
        row_count = len(rows) if (success and rows is not None) else None

        self._prev_queries.append(query)
        self._cumulative_reward += reward_obj.total
        self._best_reward = max(self._best_reward, reward_obj.total)

        # Episode ends: perfect score, or max steps reached
        solved = reward_obj.result_correct > 0 and reward_obj.total >= 0.95
        self._done = solved or (self._step >= self._task.max_steps)

        obs = self._make_observation(
            current_query=query,
            error_message=error_msg,
            execution_time_ms=elapsed_ms if success else None,
            result_preview=preview,
            result_row_count=row_count,
        )

        info: Dict[str, Any] = {
            "reward_breakdown": reward_obj.model_dump(),
            "solved": solved,
            "steps_remaining": self._task.max_steps - self._step,
            "cumulative_reward": self._cumulative_reward,
        }

        return StepResult(
            observation=obs,
            reward=reward_obj.total,
            reward_breakdown=reward_obj,
            done=self._done,
            info=info,
        )

    def state(self) -> StateResult:
        """Return a snapshot of current internal environment state."""
        return StateResult(
            task_id=self.task_id,
            step_number=self._step,
            max_steps=self._task.max_steps,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            previous_queries=list(self._prev_queries),
            best_reward_so_far=self._best_reward,
        )

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _make_observation(
        self,
        current_query: str,
        error_message: Optional[str],
        execution_time_ms: Optional[float],
        result_preview: Optional[str],
        result_row_count: Optional[int],
    ) -> SQLObservation:
        sample_data = _get_sample_data(self._conn, self._task.schema_ddl)
        return SQLObservation(
            task_id=self.task_id,
            task_description=self._task.description,
            schema_ddl=self._task.schema_ddl,
            sample_data=sample_data,
            current_query=current_query,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            result_preview=result_preview,
            result_row_count=result_row_count,
            step_number=self._step,
            max_steps=self._task.max_steps,
            hint=self._task.hint,
        )
