"""Offline tests: read-only validation, <think> stripping, execution guardrails.

Run with `python test_offline.py`. No pytest and no Groq key — the LLM is never
called. Needs Neo4j reachable for the execution tests.
"""

import sys

from agent import _strip_reasoning
from graph import HetionetGraph
from validate import UnsafeQueryError, validate

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  pass  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def test_validate() -> None:
    print("\nread-only validation")
    cases = [
        ("MATCH (n) RETURN n LIMIT 5", True),
        ("CREATE (n:Foo) RETURN n", False),
        ("MATCH (n) DETACH DELETE n", False),
        ("MATCH (n) SET n.x = 1", False),
        ("MATCH (n) REMOVE n.x RETURN n", False),
        ("LOAD CSV FROM 'x' AS row RETURN row", False),
        # Literals are blanked before scanning, so these are not false positives.
        ("MATCH (d:Disease) WHERE d.name = 'heart set attack' RETURN d LIMIT 5", True),
        ("MATCH (d) WHERE d.name CONTAINS 'create' RETURN d LIMIT 1", True),
        ("CALL dbms.components() YIELD name RETURN name", False),
        ("CALL db.labels()", True),
        ("", False),
    ]
    for cypher, expected in cases:
        try:
            validate(cypher)
            allowed = True
        except UnsafeQueryError:
            allowed = False
        check(f"{'allow' if expected else 'reject'}: {cypher[:52]!r}", allowed == expected)


def test_strip_reasoning() -> None:
    print("\nreasoning-model <think> stripping")
    cases = [
        ("<think>reasoning</think>Final answer.", "Final answer."),
        ("<think>multi\nline</think>\n\nThe answer is X.", "The answer is X."),
        ("No reasoning at all.", "No reasoning at all."),
        ("<THINK>upper</THINK>Answer", "Answer"),
        # Cut off mid-reasoning: fall back to the last paragraph.
        ("<think>unterminated\nThe real answer.", "The real answer."),
    ]
    for text, expected in cases:
        check(f"{text[:34]!r}", _strip_reasoning(text) == expected, f"got {_strip_reasoning(text)!r}")


def test_execution(graph: HetionetGraph) -> None:
    print("\nexecution guardrails")
    rows = graph.query("MATCH (d:Disease) RETURN d.name AS n LIMIT 100")
    check("row cap overrides the query's own LIMIT", len(rows) == 25, f"got {len(rows)}")

    try:
        graph.explain("MATCH (d:Disease RETURN d")
        check("EXPLAIN rejects bad syntax", False, "no error raised")
    except Exception:
        check("EXPLAIN rejects bad syntax", True)

    check("schema lists all 11 labels", len(graph.labels) == 11, f"got {len(graph.labels)}")


def main() -> int:
    test_validate()
    test_strip_reasoning()
    with HetionetGraph() as graph:
        test_execution(graph)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all offline tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
