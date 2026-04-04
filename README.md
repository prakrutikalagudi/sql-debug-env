# SQLDebugEnv

**An OpenEnv-compliant environment for training and evaluating AI agents on real-world SQL debugging tasks.**

SQL debugging is something every data engineer, analyst, and developer does daily. This environment puts an agent in the role of a SQL engineer handed a broken query — and rewards it for finding and fixing the problem.

---

## Why This Environment

SQL bugs are common, consequential, and come in distinct flavours:
- **Syntax errors** — the query fails immediately with a clear error message
- **Performance problems** — the query is correct but unacceptably slow
- **Logical bugs** — the query runs and returns results, but the results are silently wrong

Each flavour requires different reasoning. This environment covers all three, with shaped rewards that signal *partial progress* at every step — not just binary success.

---

## Environment Description

An in-process **SQLite** database is seeded with a realistic schema and data. The agent receives a broken query and must iteratively fix it by submitting SQL. Each submission is executed against the live database and graded across four signals: syntax validity, execution success, result correctness, and query efficiency.

The environment is fully stateless between episodes and deterministic — the same query always produces the same reward.

---

## Tasks

### Task 1 — `syntax_fix` (Easy)
**Target difficulty:** Frontier models should solve in 1–2 steps.

The agent receives a customer orders aggregation query with a missing comma in the `SELECT` clause. The task tests whether the agent can read an error message, identify the broken token, and apply the fix.

**Schema:** `customers`, `orders`  
**Expected correct output:** 4 rows — each customer's name and their total order spend, descending.

### Task 2 — `query_optimize` (Medium)
**Target difficulty:** Requires reasoning about query plans, not just syntax.

The agent receives a correlated subquery that returns correct results but is O(n²). The agent must rewrite it as a `JOIN` + `GROUP BY` aggregation. Correctness alone is not enough — the rewritten query must also execute under 200ms.

**Schema:** `products`, `order_items`  
**Expected correct output:** 20 rows — each product's name and its highest single-order quantity, descending.

### Task 3 — `logic_bug` (Hard)
**Target difficulty:** Requires understanding data relationships, not just SQL mechanics.

The agent receives a finance report query that runs without errors but returns inflated revenue totals. The bug is a `JOIN` on a table that has multiple rows per sales rep, causing each sale to be counted N times. The agent must identify which JOIN is incorrect and remove it — without breaking the query structure.

**Schema:** `sales_reps`, `regions`, `sales`, `rep_targets`  
**Expected correct output:** 3 rows — each sales rep's name, region, and correct total revenue.

---

## Action & Observation Spaces

### Observation (`SQLObservation`)

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Task identifier |
| `task_description` | str | Natural language task description |
| `schema_ddl` | str | Full `CREATE TABLE` DDL |
| `sample_data` | str | 3 sample rows per table |
| `current_query` | str | The current (possibly broken) query |
| `error_message` | str \| null | SQL error from last execution |
| `execution_time_ms` | float \| null | Last execution time in ms |
| `result_preview` | str \| null | First 5 rows of result |
| `result_row_count` | int \| null | Total rows returned |
| `step_number` | int | Current step (0 = initial) |
| `max_steps` | int | Episode step limit |
| `hint` | str \| null | Optional task-level hint |

### Action (`SQLAction`)

| Field | Type | Description |
|---|---|---|
| `sql_query` | str | The agent's SQL query attempt |
| `explanation` | str | Agent's reasoning (optional, not graded) |

---

## Reward Function

Each step returns a shaped reward in `[0.0, 1.0]` composed of:

| Signal | Value | Condition |
|---|---|---|
| Syntax valid | +0.10 – +0.20 | Query parses without syntax error |
| Executes | +0.15 – +0.30 | Query runs without runtime error |
| Result correct | +0.30 – +0.55 | Output matches gold result set (order-independent, float-rounded) |
| Efficient | +0.20 – +0.25 | Executes within per-task time threshold |
| Loop penalty | −0.05 per repeat | Same query submitted twice in one episode |
| Destructive penalty | −0.30 | `DROP` / `DELETE` / `TRUNCATE` attempted |

Weights differ by task. Task 3 (`logic_bug`) weights correctness at 0.55 since the broken query already executes — the entire challenge is semantic.

---

## Baseline Scores

Measured using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Avg Score | Typical Steps to Solve |
|---|---|---|---|
| `syntax_fix` | Easy | ~0.90 | 1–2 |
| `query_optimize` | Medium | ~0.60 | 2–4 |
| `logic_bug` | Hard | ~0.30 | 4–8 |

---

## Setup & Usage

### Local (Python)

```bash
git clone <repo_url>
cd sql-debug-env

pip install -r requirements.txt

# Run the server
uvicorn env.server:app --host 0.0.0.0 --port 7860

# Run tests
pytest tests/ -v

# Run baseline inference (requires API key)
export HF_TOKEN=your_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
docker build -t sql-debug-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_key \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  sql-debug-env
```

### API Reference

```bash
# List tasks
curl http://localhost:7860/tasks

# Reset (start episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "syntax_fix"}'

# Step (submit a query)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "syntax_fix",
    "sql_query": "SELECT c.name, SUM(o.amount) AS total FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id ORDER BY total DESC",
    "explanation": "Added missing comma"
  }'

# State
curl "http://localhost:7860/state?task_id=syntax_fix"
```

### Run a single task

```bash
SQL_DEBUG_TASK=logic_bug python inference.py
```

---

## Project Structure

```
sql-debug-env/
├── Dockerfile
├── openenv.yaml
├── inference.py          # Mandatory baseline inference script
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── environment.py    # SQLDebugEnv core class
│   ├── models.py         # Pydantic typed models
│   ├── tasks.py          # Task definitions (schema, seed, broken+gold queries)
│   ├── graders.py        # Deterministic grading logic
│   └── server.py         # FastAPI REST server
└── tests/
    └── test_graders.py   # 25 unit tests
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes (inference) | LLM endpoint URL |
| `MODEL_NAME` | Yes (inference) | Model identifier |
| `HF_TOKEN` | Yes (inference) | HuggingFace / API key |
| `SQL_DEBUG_TASK` | No | Override to run a single task |
