"""
Deterministic graders for SQLDebugEnv.

Each grader takes the agent's query execution results and the gold results,
and returns a shaped SQLReward with total strictly in (0, 1).
"""

import re
import time
import sqlite3
from typing import Any, List, Optional, Tuple

from env.models import SQLReward


# Small epsilon to keep score strictly between (0,1)
EPS = 0.001  

def clamp_score(value: float) -> float:
    """Ensure score is strictly between (0, 1) — well clear of boundaries."""
    return max(EPS, min(1.0 - EPS, float(value)))

# ──────────────────────────────────────────────────────────────────
# SQL execution helpers
# ──────────────────────────────────────────────────────────────────

DESTRUCTIVE_PATTERN = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def is_destructive(query: str) -> bool:
    return bool(DESTRUCTIVE_PATTERN.search(query))


def execute_query(
    conn: sqlite3.Connection, query: str
) -> Tuple[bool, Optional[List[Any]], Optional[str], float]:
    start = time.perf_counter()
    try:
        cur = conn.execute(query)
        rows = cur.fetchall()
        elapsed = (time.perf_counter() - start) * 1000
        return True, rows, None, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return False, None, str(exc), elapsed


def rows_to_comparable(rows: List[Any]) -> List[tuple]:
    normalized = []
    for row in rows:
        norm_row = tuple(
            round(float(v), 4) if isinstance(v, float) else v for v in row
        )
        normalized.append(norm_row)
    return sorted(normalized)


def results_match(agent_rows: List[Any], gold_rows: List[Any]) -> bool:
    if agent_rows is None or gold_rows is None:
        return False
    if len(agent_rows) != len(gold_rows):
        return False
    return rows_to_comparable(agent_rows) == rows_to_comparable(gold_rows)


# ──────────────────────────────────────────────────────────────────
# Shared grading logic
# ──────────────────────────────────────────────────────────────────

def grade(
    conn: sqlite3.Connection,
    agent_query: str,
    gold_rows: List[Any],
    time_threshold_ms: float,
    prev_queries: List[str],
    syntax_weight: float = 0.2,
    execute_weight: float = 0.3,
    correct_weight: float = 0.3,
    efficiency_weight: float = 0.2,
    loop_penalty: float = -0.05,
    destructive_penalty: float = -0.3,
) -> SQLReward:

    reward = SQLReward(total=EPS)  # never start at 0

    # ───── Destructive check ─────
    if is_destructive(agent_query):
        reward.destructive_penalty = destructive_penalty
        reward.total = clamp_score(destructive_penalty)
        return reward

    # ───── Loop penalty ─────
    if agent_query.strip() in [q.strip() for q in prev_queries]:
        reward.loop_penalty = loop_penalty

    # ───── Syntax check ─────
    try:
        conn.execute(f"EXPLAIN {agent_query}")
        reward.syntax_valid = syntax_weight
    except Exception:
        raw = reward.loop_penalty + reward.destructive_penalty
        reward.total = clamp_score(raw)
        return reward

    # ───── Execution check ─────
    success, agent_rows, error_msg, elapsed_ms = execute_query(conn, agent_query)
    if not success:
        raw = reward.syntax_valid + reward.loop_penalty + reward.destructive_penalty
        reward.total = clamp_score(raw)
        return reward

    reward.executes = execute_weight

    # ───── Correctness ─────
    if results_match(agent_rows, gold_rows):
        reward.result_correct = correct_weight

    # ───── Efficiency ─────
    if elapsed_ms <= time_threshold_ms:
        reward.efficient = efficiency_weight

    # ───── Final score ─────
    raw = (
        reward.syntax_valid
        + reward.executes
        + reward.result_correct
        + reward.efficient
        + reward.loop_penalty
        + reward.destructive_penalty
    )

    reward.total = clamp_score(raw)
    return reward


# ──────────────────────────────────────────────────────────────────
# Task-specific graders
# ──────────────────────────────────────────────────────────────────

def grade_syntax_fix(conn, agent_query, gold_rows, prev_queries) -> SQLReward:
    return grade(
        conn=conn,
        agent_query=agent_query,
        gold_rows=gold_rows,
        time_threshold_ms=500.0,
        prev_queries=prev_queries,
        syntax_weight=0.2,
        execute_weight=0.3,
        correct_weight=0.3,
        efficiency_weight=0.2,
    )


def grade_query_optimize(conn, agent_query, gold_rows, prev_queries) -> SQLReward:
    return grade(
        conn=conn,
        agent_query=agent_query,
        gold_rows=gold_rows,
        time_threshold_ms=200.0,
        prev_queries=prev_queries,
        syntax_weight=0.15,
        execute_weight=0.25,
        correct_weight=0.35,
        efficiency_weight=0.25,
    )


def grade_logic_bug(conn, agent_query, gold_rows, prev_queries) -> SQLReward:
    return grade(
        conn=conn,
        agent_query=agent_query,
        gold_rows=gold_rows,
        time_threshold_ms=500.0,
        prev_queries=prev_queries,
        syntax_weight=0.1,
        execute_weight=0.15,
        correct_weight=0.55,
        efficiency_weight=0.2,
    )


GRADER_MAP = {
    "syntax_fix": grade_syntax_fix,
    "query_optimize": grade_query_optimize,
    "logic_bug": grade_logic_bug,
}
