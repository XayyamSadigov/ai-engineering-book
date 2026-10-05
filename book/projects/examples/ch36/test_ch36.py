# path: book/projects/examples/ch36/test_ch36.py
"""Tests for the SQL guard. Every test runs against both engines so that the
regex fallback is held to the same contract as the sqlglot engine."""
from __future__ import annotations

import pytest

from sql_guard import HAVE_SQLGLOT, GuardConfig, SqlGuard, guard_sql, strip_string_literals

ENGINES = ["regex"] + (["sqlglot"] if HAVE_SQLGLOT else [])

ALLOWED = {"fact_orders", "dim_customer", "dim_date", "metrics.daily_revenue"}


@pytest.fixture(params=ENGINES)
def guard(request: pytest.FixtureRequest) -> SqlGuard:
    cfg = GuardConfig(
        allowed_tables=ALLOWED,
        default_limit=200,
        max_limit=1000,
        blocked_columns={"ssn", "email"},
    )
    return SqlGuard(cfg, engine=request.param)


# -- happy paths ------------------------------------------------------------
def test_adds_default_limit(guard: SqlGuard) -> None:
    r = guard.check("SELECT customer_id, SUM(amount) FROM fact_orders GROUP BY customer_id")
    assert r.ok, r.violations
    assert r.sql.rstrip().upper().endswith("LIMIT 200")
    assert not r.sql.rstrip().endswith(";")
    assert r.tables == ["fact_orders"]


def test_keeps_limit_within_max(guard: SqlGuard) -> None:
    r = guard.check("SELECT order_id FROM fact_orders LIMIT 50")
    assert r.ok
    assert "LIMIT 50" in r.sql.upper()


def test_clamps_limit_above_max(guard: SqlGuard) -> None:
    r = guard.check("SELECT order_id FROM fact_orders LIMIT 999999;")
    assert r.ok
    assert "LIMIT 1000" in r.sql.upper()
    assert "999999" not in r.sql


def test_schema_qualified_table_allowed(guard: SqlGuard) -> None:
    r = guard.check("SELECT day, revenue FROM metrics.daily_revenue WHERE day >= '2026-01-01'")
    assert r.ok, r.violations
    assert r.tables == ["metrics.daily_revenue"]


def test_cte_name_is_not_a_table(guard: SqlGuard) -> None:
    sql = """
    WITH recent AS (
        SELECT customer_id, amount FROM fact_orders WHERE order_date > '2026-01-01'
    ), ranked AS (
        SELECT customer_id, SUM(amount) AS total FROM recent GROUP BY customer_id
    )
    SELECT c.name, r.total FROM ranked r JOIN dim_customer c ON c.customer_id = r.customer_id
    """
    r = guard.check(sql)
    assert r.ok, r.violations
    assert r.tables == ["dim_customer", "fact_orders"]


def test_join_between_allowed_tables(guard: SqlGuard) -> None:
    sql = (
        "SELECT d.month, SUM(o.amount) FROM fact_orders o "
        "JOIN dim_date d ON d.date_key = o.date_key GROUP BY d.month"
    )
    r = guard.check(sql)
    assert r.ok, r.violations
    assert r.tables == ["dim_date", "fact_orders"]


def test_string_literals_do_not_trigger_false_positives(guard: SqlGuard) -> None:
    sql = "SELECT order_id FROM fact_orders WHERE note = '; DROP TABLE x -- not a comment'"
    r = guard.check(sql)
    assert r.ok, r.violations


def test_strip_string_literals_handles_escaped_quotes() -> None:
    assert strip_string_literals("SELECT 'it''s; fine' FROM t") == "SELECT '' FROM t"


# -- rejections -------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO fact_orders (order_id) VALUES (1)",
        "UPDATE fact_orders SET amount = 0",
        "DELETE FROM fact_orders",
        "DROP TABLE fact_orders",
        "CREATE TABLE t AS SELECT * FROM fact_orders",
        "TRUNCATE fact_orders",
        "GRANT SELECT ON fact_orders TO public",
    ],
)
def test_write_statements_rejected(guard: SqlGuard, sql: str) -> None:
    r = guard.check(sql)
    assert not r.ok
    assert r.codes & {"not_select", "not_read_only", "parse_error"}


def test_multiple_statements_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT 1 FROM fact_orders; DROP TABLE fact_orders")
    assert not r.ok
    assert "multiple_statements" in r.codes


def test_trailing_semicolon_is_fine(guard: SqlGuard) -> None:
    r = guard.check("SELECT order_id FROM fact_orders;")
    assert r.ok, r.violations


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT order_id FROM fact_orders -- hide the rest",
        "SELECT order_id /* sneaky */ FROM fact_orders",
    ],
)
def test_comments_rejected(guard: SqlGuard, sql: str) -> None:
    r = guard.check(sql)
    assert not r.ok
    assert "comment" in r.codes


def test_table_outside_allowlist_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT * FROM hr.salaries")
    assert not r.ok
    assert "table_not_allowed" in r.codes


def test_exfiltration_join_rejected(guard: SqlGuard) -> None:
    sql = (
        "SELECT o.order_id, s.salary FROM fact_orders o "
        "JOIN hr.salaries s ON s.employee_id = o.sales_rep_id"
    )
    r = guard.check(sql)
    assert not r.ok
    assert "table_not_allowed" in r.codes
    assert "hr.salaries" in r.tables


def test_subquery_table_checked(guard: SqlGuard) -> None:
    sql = "SELECT order_id FROM fact_orders WHERE customer_id IN (SELECT id FROM secret_list)"
    r = guard.check(sql)
    assert not r.ok
    assert "table_not_allowed" in r.codes


def test_cte_cannot_smuggle_a_table(guard: SqlGuard) -> None:
    sql = "WITH x AS (SELECT * FROM hr.salaries) SELECT * FROM x"
    r = guard.check(sql)
    assert not r.ok
    assert "table_not_allowed" in r.codes


def test_forbidden_function_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT order_id, pg_sleep(10) FROM fact_orders")
    assert not r.ok
    assert "forbidden_function" in r.codes


def test_blocked_pii_column_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT email FROM dim_customer")
    assert not r.ok
    assert "blocked_column" in r.codes


def test_blocked_column_in_where_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT name FROM dim_customer WHERE ssn = '123'")
    assert not r.ok
    assert "blocked_column" in r.codes


def test_select_star_optionally_forbidden() -> None:
    for engine in ENGINES:
        cfg = GuardConfig(allowed_tables=ALLOWED, forbid_select_star=True)
        r = SqlGuard(cfg, engine=engine).check("SELECT * FROM fact_orders")
        assert not r.ok, engine
        assert "select_star" in r.codes, engine


def test_for_update_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT order_id FROM fact_orders FOR UPDATE")
    assert not r.ok
    assert "not_read_only" in r.codes


def test_select_into_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT order_id INTO copied FROM fact_orders")
    assert not r.ok
    assert r.codes & {"not_read_only", "parse_error", "table_not_allowed"}


def test_table_function_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT * FROM generate_series(1, 1000000)")
    assert not r.ok
    assert r.codes & {"table_function", "table_not_allowed"}


def test_non_literal_limit_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELECT order_id FROM fact_orders LIMIT ALL")
    assert not r.ok
    assert r.codes & {"limit_not_literal", "parse_error"}


def test_garbage_rejected(guard: SqlGuard) -> None:
    r = guard.check("SELEC order_id FORM fact_orders")
    assert not r.ok


def test_empty_rejected(guard: SqlGuard) -> None:
    r = guard.check("   ")
    assert not r.ok
    assert "empty" in r.codes


# -- wiring -------------------------------------------------------------------
def test_auto_engine_prefers_sqlglot() -> None:
    g = SqlGuard(GuardConfig(allowed_tables=ALLOWED))
    assert g.engine == ("sqlglot" if HAVE_SQLGLOT else "regex")


def test_convenience_wrapper_lowercases_allowlist() -> None:
    r = guard_sql("SELECT order_id FROM Fact_Orders", allowed_tables={"FACT_ORDERS"})
    assert r.ok, r.violations


@pytest.mark.skipif(not HAVE_SQLGLOT, reason="sqlglot not installed")
def test_sqlglot_renders_in_configured_dialect() -> None:
    cfg = GuardConfig(allowed_tables=ALLOWED, dialect="postgres")
    r = SqlGuard(cfg, engine="sqlglot").check("SELECT order_id FROM fact_orders")
    assert r.ok
    assert r.sql == "SELECT order_id FROM fact_orders LIMIT 200"
