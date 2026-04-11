"""
inference.py — SQLDebugEnv Baseline Inference Script
=====================================================

Runs an LLM agent against all 3 tasks using the OpenAI client.
Emits mandatory [START] / [STEP] / [END] stdout format.

Environment variables:
    API_BASE_URL      LLM endpoint (default: HuggingFace router)
    MODEL_NAME        Model identifier
    HF_TOKEN          API key
    SQL_DEBUG_TASK    Override to run a single task (optional)
"""

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Add project root to path so env/ is importable ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

from env.environment import SQLDebugEnv
from env.models import SQLAction
from env.tasks import TASKS

# ── Config ──────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "sql-debug-env"
TEMPERATURE  = 0.2
MAX_TOKENS   = 512
SUCCESS_THRESHOLD = 0.7   # reward >= this → episode is "successful"

# Small epsilon — scores must be strictly inside (0, 1)
EPS = 0.001 

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SQL engineer. You will be given a broken SQL query and a database schema.
Your job is to fix the query so it:
1. Runs without errors
2. Returns the correct result set
3. Is efficient (avoids unnecessary subqueries or repeated scans)

Reply with ONLY a JSON object in this exact format (no markdown, no extra text):
{
  "sql_query": "<your fixed SQL here>",
  "explanation": "<one sentence describing what you changed and why>"
}
""").strip()


# ── Mandatory log helpers ────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe_action = action.replace("\n", " ").replace("\r", " ")[:120]
    error_val = error.replace("\n", " ")[:80] if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.6f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.6f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────

def build_user_prompt(obs) -> str:
    history_note = f"(step {obs.step_number} of {obs.max_steps})"
    error_block = f"\nError from last attempt:\n{obs.error_message}" if obs.error_message else ""
    preview_block = f"\nLast result preview:\n{obs.result_preview}" if obs.result_preview else ""
    time_block = f"\nLast execution time: {obs.execution_time_ms:.1f}ms" if obs.execution_time_ms else ""
    return textwrap.dedent(f"""
        Task {history_note}: {obs.task_description}

        Schema:
        {obs.schema_ddl}

        Sample data:
        {obs.sample_data}

        Current query (may be broken):
        {obs.current_query}
        {error_block}{preview_block}{time_block}

        Hint: {obs.hint or 'None'}

        Fix the SQL query. Reply with JSON only.
    """).strip()


def call_model(client: OpenAI, obs) -> SQLAction:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if model wraps in ```json
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        import json
        parsed = json.loads(raw)
        return SQLAction(
            sql_query=parsed.get("sql_query", obs.current_query),
            explanation=parsed.get("explanation", ""),
        )
    except Exception as exc:
        print(f"[DEBUG] Model call failed: {exc}", flush=True)
        # Fall back: return current query unchanged
        return SQLAction(sql_query=obs.current_query, explanation="fallback")


# ── Episode runner ────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> dict:
    env = SQLDebugEnv(task_id=task_id)

    rewards: List[float] = []
    steps_taken = 0
    score = EPS          # ← default: never 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = env.reset()
        obs = reset_result.observation

        task = TASKS[task_id]
        max_steps = task.max_steps

        for step in range(1, max_steps + 1):
            action = call_model(client, obs)
            step_result = env.step(action)

            reward = step_result.reward
            done = step_result.done
            error = step_result.observation.error_message

            rewards.append(reward)
            steps_taken = step
            obs = step_result.observation

            log_step(
                step=step,
                action=action.sql_query,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Best single-step reward — clamped strictly inside (0, 1)
        raw_score = max(rewards) if rewards else EPS
        score = max(0.001, min(0.999, float(raw_score))) #← FIXED: never 0.0 or 1.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = EPS   # ← FIXED: never 0.0 on exception path

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    single_task = os.getenv("SQL_DEBUG_TASK", "")
    task_ids = [single_task] if single_task in TASKS else list(TASKS.keys())

    results = []
    for task_id in task_ids:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run_episode(client, task_id)
        results.append(result)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"[{status}] {r['task_id']:20s}  score={r['score']:.6f}  steps={r['steps']}", flush=True)

    avg = sum(r["score"] for r in results) / len(results) if results else EPS
    print(f"\nAverage score: {avg:.6f}", flush=True)


if __name__ == "__main__":
    main()
