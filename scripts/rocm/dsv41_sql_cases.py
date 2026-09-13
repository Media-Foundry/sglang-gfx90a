"""Fixed real coding tasks with independent Python oracles; no model execution."""

import random


CASES = {
    "latest_build": {
        "schema": "CREATE TABLE builds(id INTEGER, repo TEXT, started_at INTEGER, status TEXT);",
        "table": "builds",
        "spec": "Return (repo, id, started_at) for the latest SUCCESSFUL build of each repo. "
                "status must equal 'success'. Latest means greatest started_at, then greatest id. "
                "A newer failed build must not hide an older success. Omit repos without success. "
                "Order by repo. Timestamps are integer seconds and can tie.",
    },
    "event_sessions": {
        "schema": "CREATE TABLE events(id INTEGER, user_id TEXT, ts INTEGER);",
        "table": "events",
        "spec": "Sessionize events independently per user ordered by (ts,id). A new session starts "
                "only when the gap from the previous event is STRICTLY GREATER THAN 1800 seconds. "
                "Equal timestamps and gap=1800 stay in the same session. Number sessions from1 "
                "per user. Return (user_id, session_no, start_ts, end_ts, event_count), ordered "
                "by user_id then session_no. Empty input produces no rows.",
    },
    "usage_dedup": {
        "schema": "CREATE TABLE attempts(id INTEGER, request_id TEXT, user_id TEXT, updated_at INTEGER, "
                  "status TEXT, input_tokens INTEGER, output_tokens INTEGER);",
        "table": "attempts",
        "spec": "Each request_id has one owner user and may have multiple attempt records. Choose "
                "the latest record per request by greatest updated_at then greatest id BEFORE "
                "filtering status. Charge only if this latest record's status='complete'. "
                "NULL token counts mean zero. Return (user_id,total_input_tokens,total_output_tokens) "
                "for EVERY user appearing in attempts, including users whose totals are zero. "
                "Order by user_id. Empty input produces no rows.",
    },
    "dependency_closure": {
        "schema": "CREATE TABLE dependencies(module TEXT, requires TEXT);",
        "table": "dependencies",
        "spec": "Find all modules transitively required by root module 'app', where each row "
                "means module depends on requires. Return one column named dependency, sorted "
                "lexicographically, unique, excluding 'app' itself. Edges may be duplicated, "
                "cyclic or self-referential; the query MUST terminate. Include a dependency even "
                "if it never appears in the module column. Root with no outgoing edges returns no rows.",
    },
}


def prompt_for(name):
    task = CASES[name]
    return (
        "You are reviewing a production service's data pipeline. Implement the following task "
        "as ONE read-only SQLite3 query (CTEs and window functions are available).\n\n"
        + task["schema"] + "\n\n" + task["spec"]
        + "\n\nFirst explain your design, correctness at boundaries, and two likely implementation "
        "mistakes in 300-450 English words. Then give exactly one fenced ```sql code block "
        "containing the complete query, no setup or schema changes. Do not give multiple query "
        "alternatives. No external functions, files, extensions, or other tables."
    )


def fixture(name, seed):
    rng = random.Random(seed)
    if name == "latest_build":
        rows = [(i, rng.choice(["api", "worker", "ui"]), rng.choice([0, 10, 20, 30]),
                 rng.choice(["success", "failed", "running"])) for i in range(1, 50)]
        # Always exercise a later failure and equal-time successful tie.
        rows += [(100, "ties", 4, "success"), (101, "ties", 4, "success"),
                 (102, "ties", 5, "failed"), (103, "no_success", 9, "failed")]
        if seed == 0:
            rows = []
        chosen = {}
        for rid, repo, ts, status in rows:
            if status == "success" and (ts, rid) > chosen.get(repo, (-1, -1)):
                chosen[repo] = (ts, rid)
        expected = [(repo, rid, ts) for repo, (ts, rid) in sorted(chosen.items())]
    elif name == "event_sessions":
        rows = []
        for user in ["alice", "bob", "carol"]:
            ts = 0
            for _ in range(18):
                ts += rng.choice([0, 1, 1799, 1800, 1801, 3600])
                rows.append((len(rows) + 1, user, ts))
        if seed == 0:
            rows = []
        expected = []
        for user in sorted({r[1] for r in rows}):
            times = sorted((ts, rid) for rid, u, ts in rows if u == user)
            session, begin, end, count = 0, None, None, 0
            for ts, _ in times:
                if end is None or ts - end > 1800:
                    if count:
                        expected.append((user, session, begin, end, count))
                    session, begin, count = session + 1, ts, 0
                end, count = ts, count + 1
            if count:
                expected.append((user, session, begin, end, count))
    elif name == "usage_dedup":
        rows = []
        for request in range(20):
            user = ["alice", "bob", "carol"][request % 3]
            for _ in range(rng.randint(1, 4)):
                rows.append((len(rows) + 1, str(request), user, rng.choice([1, 2, 3]),
                             rng.choice(["complete", "failed", "cancelled"]),
                             rng.choice([None, 0, 7, 100]), rng.choice([None, 0, 13, 20])))
        rows += [(1000, "cancel", "zero", 2, "complete", 90, 10),
                 (1001, "cancel", "zero", 2, "cancelled", 90, 10)]
        if seed == 0:
            rows = []
        totals = {r[2]: [0, 0] for r in rows}
        latest = {}
        for row in rows:
            key = row[1]
            if key not in latest or (row[3], row[0]) > (latest[key][3], latest[key][0]):
                latest[key] = row
        for _, _, user, _, status, inp, out in latest.values():
            if status == "complete":
                totals[user][0] += inp or 0
                totals[user][1] += out or 0
        expected = [(user, *counts) for user, counts in sorted(totals.items())]
    elif name == "dependency_closure":
        nodes = ["app", "api", "db", "cache", "lib", "leaf", "isolated"]
        rows = [(rng.choice(nodes[:-1]), rng.choice(nodes)) for _ in range(15)]
        rows += [("app", "api"), ("app", "api"), ("api", "db"), ("db", "api"),
                 ("db", "app"), ("app", "app")]
        if seed == 0:
            rows = [("isolated", "leaf")]
        seen, todo = set(), ["app"]
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo.extend(target for source, target in rows if source == node)
        expected = [(node,) for node in sorted(seen - {"app"})]
    else:
        raise ValueError(name)
    rng.shuffle(rows)
    return rows, expected
