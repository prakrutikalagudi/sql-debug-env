"""
Task definitions for SQLDebugEnv.

Task 1 — Easy:   Fix a syntax error in a simple SELECT query
Task 2 — Medium: Optimize a slow N+1 / subquery pattern
Task 3 — Hard:   Fix a subtle logical bug causing wrong results (silent duplicates)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Task:
    task_id: str
    description: str
    schema_ddl: str          # CREATE TABLE statements
    seed_sql: List[str]      # INSERT statements to populate the DB
    broken_query: str        # The broken query handed to the agent
    gold_query: str          # Reference correct query (used to compute gold results)
    hint: Optional[str]
    max_steps: int
    time_threshold_ms: float # Execution must be under this for efficiency reward
    difficulty: str


TASKS: dict[str, Task] = {}


# ──────────────────────────────────────────────
# TASK 1 — Easy: Fix a syntax error
# ──────────────────────────────────────────────
TASKS["syntax_fix"] = Task(
    task_id="syntax_fix",
    difficulty="easy",
    description=(
        "The analytics team has a broken SQL query. "
        "It should return each customer's name and their total order amount, "
        "but it fails with a syntax error. Fix it so it runs and returns the correct results."
    ),
    schema_ddl="""
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    country     TEXT NOT NULL
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    amount      REAL    NOT NULL,
    order_date  TEXT    NOT NULL
);
""".strip(),
    seed_sql=[
        "INSERT INTO customers VALUES (1, 'Alice Johnson', 'alice@example.com', 'US');",
        "INSERT INTO customers VALUES (2, 'Bob Smith', 'bob@example.com', 'UK');",
        "INSERT INTO customers VALUES (3, 'Carol White', 'carol@example.com', 'US');",
        "INSERT INTO customers VALUES (4, 'Dan Brown', 'dan@example.com', 'CA');",
        "INSERT INTO orders VALUES (1, 1, 250.00, '2024-01-10');",
        "INSERT INTO orders VALUES (2, 1, 180.50, '2024-02-15');",
        "INSERT INTO orders VALUES (3, 2, 320.00, '2024-01-20');",
        "INSERT INTO orders VALUES (4, 3, 95.75,  '2024-03-01');",
        "INSERT INTO orders VALUES (5, 3, 430.00, '2024-03-15');",
        "INSERT INTO orders VALUES (6, 4, 200.00, '2024-02-28');",
    ],
    # Broken: missing comma between c.name and SUM(o.amount)
    broken_query="""
SELECT c.name
       SUM(o.amount) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC;
""".strip(),
    gold_query="""
SELECT c.name, SUM(o.amount) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC;
""".strip(),
    hint="Look at the SELECT clause carefully — are all the columns properly separated?",
    max_steps=6,
    time_threshold_ms=500.0,
)


# ──────────────────────────────────────────────
# TASK 2 — Medium: Optimize a slow correlated subquery
# ──────────────────────────────────────────────
TASKS["query_optimize"] = Task(
    task_id="query_optimize",
    difficulty="medium",
    description=(
        "A reporting query finds each product's name and its highest single-order quantity sold. "
        "The current implementation uses a correlated subquery that is extremely slow on large tables. "
        "Rewrite it using a JOIN or aggregation so it returns the same correct results, "
        "but runs efficiently (under 200ms)."
    ),
    schema_ddl="""
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    unit_price   REAL    NOT NULL
);

CREATE TABLE order_items (
    item_id    INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity   INTEGER NOT NULL,
    sale_date  TEXT    NOT NULL
);

CREATE INDEX idx_order_items_product ON order_items(product_id);
""".strip(),
    seed_sql=(
        [f"INSERT INTO products VALUES ({i}, 'Product {i}', 'Cat{(i%3)+1}', {10+i*0.5});" for i in range(1, 21)] +
        [f"INSERT INTO order_items VALUES ({i}, {(i%20)+1}, {(i%15)+1}, '2024-0{(i%9)+1}-01');" for i in range(1, 201)]
    ),
    # Broken: uses AVG instead of MAX — runs fine but returns wrong values.
    # The correct fix is to use MAX() with a JOIN+GROUP BY for clarity and performance.
    broken_query="""
SELECT p.product_name,
       (SELECT AVG(oi2.quantity)
        FROM order_items oi2
        WHERE oi2.product_id = p.product_id) AS max_quantity_sold
FROM products p
ORDER BY max_quantity_sold DESC;
""".strip(),
    gold_query="""
SELECT p.product_name, MAX(oi.quantity) AS max_quantity_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name
ORDER BY max_quantity_sold DESC;
""".strip(),
    hint=(
        "The correlated subquery runs once per product row. "
        "A JOIN with GROUP BY aggregates in a single pass — much faster."
    ),
    max_steps=8,
    time_threshold_ms=200.0,
)


# ──────────────────────────────────────────────
# TASK 3 — Hard: Fix a silent logical bug (wrong JOIN causing duplicates)
# ──────────────────────────────────────────────
TASKS["logic_bug"] = Task(
    task_id="logic_bug",
    difficulty="hard",
    description=(
        "A finance report calculates total revenue per sales rep, joined with their region. "
        "The query runs without errors and returns rows — but the revenue figures are inflated. "
        "A sales rep's revenue is being counted multiple times due to a bad JOIN condition. "
        "Find and fix the logical bug so the totals are correct."
    ),
    schema_ddl="""
CREATE TABLE sales_reps (
    rep_id     INTEGER PRIMARY KEY,
    rep_name   TEXT    NOT NULL,
    region_id  INTEGER NOT NULL
);

CREATE TABLE regions (
    region_id   INTEGER PRIMARY KEY,
    region_name TEXT    NOT NULL,
    country     TEXT    NOT NULL
);

CREATE TABLE sales (
    sale_id  INTEGER PRIMARY KEY,
    rep_id   INTEGER NOT NULL REFERENCES sales_reps(rep_id),
    amount   REAL    NOT NULL,
    sale_date TEXT   NOT NULL
);

CREATE TABLE rep_targets (
    target_id  INTEGER PRIMARY KEY,
    rep_id     INTEGER NOT NULL REFERENCES sales_reps(rep_id),
    year       INTEGER NOT NULL,
    target_amt REAL    NOT NULL
);
""".strip(),
    seed_sql=[
        "INSERT INTO regions VALUES (1, 'North America', 'US');",
        "INSERT INTO regions VALUES (2, 'Europe', 'UK');",
        "INSERT INTO sales_reps VALUES (1, 'Sarah Connor', 1);",
        "INSERT INTO sales_reps VALUES (2, 'John Doe', 2);",
        "INSERT INTO sales_reps VALUES (3, 'Jane Smith', 1);",
        # 2 targets per rep to demonstrate the duplication bug
        "INSERT INTO rep_targets VALUES (1, 1, 2023, 50000);",
        "INSERT INTO rep_targets VALUES (2, 1, 2024, 60000);",
        "INSERT INTO rep_targets VALUES (3, 2, 2023, 45000);",
        "INSERT INTO rep_targets VALUES (4, 2, 2024, 55000);",
        "INSERT INTO rep_targets VALUES (5, 3, 2023, 40000);",
        "INSERT INTO rep_targets VALUES (6, 3, 2024, 50000);",
        "INSERT INTO sales VALUES (1, 1, 12000.00, '2024-01-15');",
        "INSERT INTO sales VALUES (2, 1, 8500.00,  '2024-02-20');",
        "INSERT INTO sales VALUES (3, 2, 21000.00, '2024-01-10');",
        "INSERT INTO sales VALUES (4, 3, 5000.00,  '2024-03-05');",
        "INSERT INTO sales VALUES (5, 3, 9000.00,  '2024-03-20');",
    ],
    # Bug: joins rep_targets WITHOUT filtering by year → each sale counted N times (once per target row)
    broken_query="""
SELECT sr.rep_name,
       r.region_name,
       SUM(s.amount) AS total_revenue
FROM sales_reps sr
JOIN regions r        ON r.region_id  = sr.region_id
JOIN sales s          ON s.rep_id     = sr.rep_id
JOIN rep_targets rt   ON rt.rep_id    = sr.rep_id
GROUP BY sr.rep_id, sr.rep_name, r.region_name
ORDER BY total_revenue DESC;
""".strip(),
    gold_query="""
SELECT sr.rep_name,
       r.region_name,
       SUM(s.amount) AS total_revenue
FROM sales_reps sr
JOIN regions r ON r.region_id = sr.region_id
JOIN sales s   ON s.rep_id    = sr.rep_id
GROUP BY sr.rep_id, sr.rep_name, r.region_name
ORDER BY total_revenue DESC;
""".strip(),
    hint=(
        "The query joins a table that has multiple rows per sales rep. "
        "Check whether every JOIN is strictly necessary for the result being computed."
    ),
    max_steps=10,
    time_threshold_ms=500.0,
)
