"""Validate independent fixtures and reject unsafe/infinite generated SQL."""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/rocm"
sys.path.insert(0, str(SCRIPTS))
try:
    from check_dsv41_sql_functional import check_query, extract_sql
finally:
    sys.path.pop(0)


REFERENCES = {
    "latest_build": """
      SELECT repo,id,started_at FROM (
        SELECT *,row_number() OVER(PARTITION BY repo ORDER BY started_at DESC,id DESC) AS rn
        FROM builds WHERE status='success'
      ) WHERE rn=1 ORDER BY repo
    """,
    "event_sessions": """
      WITH ordered AS (
        SELECT *,lag(ts) OVER(PARTITION BY user_id ORDER BY ts,id) AS previous FROM events
      ), tagged AS (
        SELECT *,sum(CASE WHEN previous IS NULL OR ts-previous>1800 THEN 1 ELSE 0 END)
          OVER(PARTITION BY user_id ORDER BY ts,id ROWS UNBOUNDED PRECEDING) AS session_no
        FROM ordered
      ) SELECT user_id,session_no,min(ts),max(ts),count(*) FROM tagged
        GROUP BY user_id,session_no ORDER BY user_id,session_no
    """,
    "usage_dedup": """
      WITH ranked AS (
        SELECT *,row_number() OVER(PARTITION BY request_id ORDER BY updated_at DESC,id DESC) AS rn
        FROM attempts
      ) SELECT user_id,
          sum(CASE WHEN rn=1 AND status='complete' THEN coalesce(input_tokens,0) ELSE 0 END),
          sum(CASE WHEN rn=1 AND status='complete' THEN coalesce(output_tokens,0) ELSE 0 END)
        FROM ranked GROUP BY user_id ORDER BY user_id
    """,
    "dependency_closure": """
      WITH RECURSIVE closure(node) AS (
        SELECT requires FROM dependencies WHERE module='app'
        UNION
        SELECT d.requires FROM dependencies d JOIN closure c ON d.module=c.node
      ) SELECT node AS dependency FROM closure WHERE node<>'app' ORDER BY node
    """,
}


@pytest.mark.parametrize("name", REFERENCES)
def test_independent_python_fixture_agrees_with_known_correct_sql(name):
    result = check_query(name, REFERENCES[name])
    assert len(result["fixtures"]) == 16
    assert result["passed"], result


@pytest.mark.parametrize("query", [
    "DROP TABLE builds", "DELETE FROM builds", "PRAGMA database_list",
    "ATTACH DATABASE '/tmp/dsv41-should-not-exist.db' AS ext",
    "SELECT load_extension('/tmp/no-extension')", "SELECT name FROM sqlite_master",
    "SELECT 1; SELECT 2",
])
def test_non_readonly_or_external_access_is_rejected(query):
    result = check_query("latest_build", query, seeds=2)
    assert not result["passed"]
    assert all(row.get("error") for row in result["fixtures"])


def test_runaway_recursion_is_interrupted():
    query = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n"
    result = check_query("latest_build", query, seeds=2, vm_callbacks=2)
    assert not result["passed"]
    assert all("interrupted" in row["error"] for row in result["fixtures"])


def test_wrong_tie_or_filtered_latest_query_is_rejected():
    wrong = REFERENCES["latest_build"].replace("FROM builds WHERE status='success'", "FROM builds").replace(
        "WHERE rn=1 ORDER BY repo", "WHERE rn=1 AND status='success' ORDER BY repo")
    assert not check_query("latest_build", wrong)["passed"]


def test_sql_fence_and_nonempty_fixture_guards():
    assert extract_sql("Explanation\n```sql\nSELECT 1;\n```") == "SELECT 1;"
    for text in ("SELECT 1", "```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```"):
        with pytest.raises(ValueError):
            extract_sql(text)
    with pytest.raises(ValueError):
        check_query("latest_build", "SELECT 1", seeds=0)
