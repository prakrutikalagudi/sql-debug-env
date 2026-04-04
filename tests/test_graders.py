"""
Tests for SQLDebugEnv graders and environment.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sqlite3
import pytest

from env.environment import SQLDebugEnv
from env.graders import execute_query, grade, is_destructive, results_match
from env.models import SQLAction
from env.tasks import TASKS


# ──────────────────────────────────────────────
# Utility tests
# ──────────────────────────────────────────────

class TestUtilities:
    def test_destructive_detection_drop(self):
        assert is_destructive("DROP TABLE users") is True

    def test_destructive_detection_delete(self):
        assert is_destructive("DELETE FROM orders WHERE id=1") is True

    def test_destructive_detection_select(self):
        assert is_destructive("SELECT * FROM users") is False

    def test_results_match_equal(self):
        a = [(1, "Alice", 100.0), (2, "Bob", 200.0)]
        b = [(2, "Bob", 200.0), (1, "Alice", 100.0)]  # different order
        assert results_match(a, b) is True

    def test_results_match_float_rounding(self):
        a = [(1, 100.0001)]
        b = [(1, 100.0001)]
        assert results_match(a, b) is True

    def test_results_match_different_counts(self):
        a = [(1,), (2,)]
        b = [(1,)]
        assert results_match(a, b) is False

    def test_results_match_none(self):
        assert results_match(None, [(1,)]) is False


# ──────────────────────────────────────────────
# Task 1 — Syntax fix
# ──────────────────────────────────────────────

class TestTask1SyntaxFix:
    def setup_method(self):
        self.env = SQLDebugEnv(task_id="syntax_fix")
        self.env.reset()

    def test_broken_query_fails(self):
        """The initial broken query should produce a syntax error."""
        task = TASKS["syntax_fix"]
        conn = sqlite3.connect(":memory:")
        conn.executescript(task.schema_ddl)
        for stmt in task.seed_sql:
            conn.execute(stmt)
        success, _, error, _ = execute_query(conn, task.broken_query)
        assert success is False
        assert error is not None

    def test_gold_query_succeeds(self):
        """The gold query should run cleanly."""
        task = TASKS["syntax_fix"]
        conn = sqlite3.connect(":memory:")
        conn.executescript(task.schema_ddl)
        for stmt in task.seed_sql:
            conn.execute(stmt)
        success, rows, _, _ = execute_query(conn, task.gold_query)
        assert success is True
        assert len(rows) == 4  # 4 customers each with orders

    def test_correct_query_gets_high_reward(self):
        result = self.env.step(SQLAction(
            sql_query="""
                SELECT c.name, SUM(o.amount) AS total_spent
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                GROUP BY c.customer_id, c.name
                ORDER BY total_spent DESC;
            """,
            explanation="Added missing comma"
        ))
        assert result.reward >= 0.8
        assert result.reward_breakdown.result_correct > 0

    def test_syntax_error_gives_only_partial_reward(self):
        result = self.env.step(SQLAction(
            sql_query="SELECT c.name SUM(o.amount) FROM customers c JOIN orders o ON c.customer_id = o.customer_id",
            explanation="Still broken"
        ))
        assert result.reward < 0.5
        assert result.reward_breakdown.syntax_valid == 0.0

    def test_destructive_query_penalized(self):
        result = self.env.step(SQLAction(sql_query="DROP TABLE orders", explanation="bad"))
        assert result.reward == 0.0
        assert result.reward_breakdown.destructive_penalty < 0

    def test_loop_penalty_on_repeat(self):
        q = "SELECT * FROM customers"
        self.env.step(SQLAction(sql_query=q, explanation="first"))
        result2 = self.env.step(SQLAction(sql_query=q, explanation="repeat"))
        assert result2.reward_breakdown.loop_penalty < 0

    def test_episode_done_at_max_steps(self):
        task = TASKS["syntax_fix"]
        for _ in range(task.max_steps):
            result = self.env.step(SQLAction(sql_query="SELECT 1", explanation="dummy"))
        assert result.done is True


# ──────────────────────────────────────────────
# Task 2 — Query optimization
# ──────────────────────────────────────────────

class TestTask2QueryOptimize:
    def setup_method(self):
        self.env = SQLDebugEnv(task_id="query_optimize")
        self.env.reset()

    def test_gold_query_returns_rows(self):
        task = TASKS["query_optimize"]
        conn = sqlite3.connect(":memory:")
        conn.executescript(task.schema_ddl)
        for stmt in task.seed_sql:
            conn.execute(stmt)
        success, rows, _, _ = execute_query(conn, task.gold_query)
        assert success is True
        assert len(rows) == 20  # 20 products

    def test_optimized_query_high_reward(self):
        result = self.env.step(SQLAction(
            sql_query="""
                SELECT p.product_name, MAX(oi.quantity) AS max_quantity_sold
                FROM products p
                JOIN order_items oi ON p.product_id = oi.product_id
                GROUP BY p.product_id, p.product_name
                ORDER BY max_quantity_sold DESC;
            """,
            explanation="Replaced correlated subquery with JOIN + GROUP BY"
        ))
        assert result.reward >= 0.7
        assert result.reward_breakdown.result_correct > 0


# ──────────────────────────────────────────────
# Task 3 — Logic bug
# ──────────────────────────────────────────────

class TestTask3LogicBug:
    def setup_method(self):
        self.env = SQLDebugEnv(task_id="logic_bug")
        self.env.reset()

    def test_broken_query_executes_but_wrong(self):
        """Broken query runs but returns wrong (inflated) totals."""
        task = TASKS["logic_bug"]
        conn = sqlite3.connect(":memory:")
        conn.executescript(task.schema_ddl)
        for stmt in task.seed_sql:
            conn.execute(stmt)

        _, broken_rows, err, _ = execute_query(conn, task.broken_query)
        _, gold_rows, _, _ = execute_query(conn, task.gold_query)

        assert err is None  # runs without error
        assert not results_match(broken_rows, gold_rows)  # but wrong results

    def test_fixed_query_correct(self):
        result = self.env.step(SQLAction(
            sql_query="""
                SELECT sr.rep_name,
                       r.region_name,
                       SUM(s.amount) AS total_revenue
                FROM sales_reps sr
                JOIN regions r ON r.region_id = sr.region_id
                JOIN sales s   ON s.rep_id    = sr.rep_id
                GROUP BY sr.rep_id, sr.rep_name, r.region_name
                ORDER BY total_revenue DESC;
            """,
            explanation="Removed unnecessary JOIN on rep_targets which caused row multiplication"
        ))
        assert result.reward_breakdown.result_correct > 0
        assert result.reward >= 0.7


# ──────────────────────────────────────────────
# Environment API tests
# ──────────────────────────────────────────────

class TestEnvironmentAPI:
    def test_reset_returns_observation(self):
        env = SQLDebugEnv(task_id="syntax_fix")
        result = env.reset()
        assert result.observation.task_id == "syntax_fix"
        assert result.observation.step_number == 0
        assert result.observation.current_query != ""

    def test_state_tracks_steps(self):
        env = SQLDebugEnv(task_id="syntax_fix")
        env.reset()
        env.step(SQLAction(sql_query="SELECT 1"))
        state = env.state()
        assert state.step_number == 1

    def test_step_after_done_raises(self):
        env = SQLDebugEnv(task_id="syntax_fix")
        env.reset()
        task = TASKS["syntax_fix"]
        for _ in range(task.max_steps):
            env.step(SQLAction(sql_query="SELECT 1"))
        with pytest.raises(RuntimeError, match="Episode is done"):
            env.step(SQLAction(sql_query="SELECT 1"))

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            SQLDebugEnv(task_id="nonexistent_task")

    def test_reset_clears_state(self):
        env = SQLDebugEnv(task_id="syntax_fix")
        env.reset()
        env.step(SQLAction(sql_query="SELECT 1"))
        env.reset()
        state = env.state()
        assert state.step_number == 0
        assert state.cumulative_reward == 0.0
        assert state.previous_queries == []

    def test_all_tasks_loadable(self):
        for task_id in TASKS:
            env = SQLDebugEnv(task_id=task_id)
            result = env.reset()
            assert result.observation.task_id == task_id

    def test_reward_always_in_range(self):
        env = SQLDebugEnv(task_id="logic_bug")
        env.reset()
        for query in ["SELECT 1", "DROP TABLE sales", "SELECT * FROM sales_reps"]:
            env2 = SQLDebugEnv(task_id="logic_bug")
            env2.reset()
            result = env2.step(SQLAction(sql_query=query))
            assert 0.0 <= result.reward <= 1.0, f"reward out of range for: {query}"
