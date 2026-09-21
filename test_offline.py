"""Offline tests: validation, <think> stripping, guardrails, tracing, scoring.

Run with `python test_offline.py`. No pytest and no Groq key — the LLM is never
called, so this costs no tokens. Needs Neo4j reachable for the execution tests.
"""

import sys

import metrics
import observability
from agent import AgentRun, _CypherTrace, _strip_reasoning
from graph import HetionetGraph
from judge import RUBRICS, Judgment, _expected_block, _truncate
from metrics import Expectation
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


def _fake_run(**overrides) -> AgentRun:
    defaults = {
        "question": "What compounds treat epilepsy syndrome?",
        "answer": "At least 25 compounds, including Clonazepam and Carbamazepine.",
        "queries": ["MATCH (c:Compound)-[:TREATS_CtD]->(d:Disease) RETURN c.name LIMIT 25"],
        "results": ['[{"compound": "Clonazepam"}]'],
        "tokens": 1200,
        "latency_s": 2.5,
    }
    return AgentRun(**(defaults | overrides))


def test_trace_callback() -> None:
    """The callback must capture queries even when the run later blows up."""
    print("\nrun tracing")
    trace = _CypherTrace()
    trace.on_tool_start({}, "MATCH (n) RETURN n", inputs={"cypher": "MATCH (n) RETURN n"})
    trace.on_tool_end("[]")
    # A tool call logged without an inputs dict must still record the query.
    trace.on_tool_start({}, "CALL db.labels()")
    trace.on_tool_end("REJECTED: nope")

    check("captures both queries", trace.queries == ["MATCH (n) RETURN n", "CALL db.labels()"])
    check("captures both results", trace.results == ["[]", "REJECTED: nope"])

    class _Response:
        llm_output = {"token_usage": {"total_tokens": 700}}

    trace.on_llm_end(_Response())
    trace.on_llm_end(_Response())
    check("sums token usage across calls", trace.tokens == 1400, f"got {trace.tokens}")

    class _NoUsage:
        llm_output = None

    trace.on_llm_end(_NoUsage())
    check("survives a response with no usage", trace.tokens == 1400)


def test_metrics() -> None:
    print("\ndeterministic metrics")
    good = metrics.deterministic(_fake_run())
    check("counts one query", good["query_count"] == 1)
    check("sees returned rows", good["returned_rows"] is True)
    check("no cypher errors", good["cypher_errors"] == 0 and good["cypher_valid_rate"] == 1.0)

    bad = metrics.deterministic(
        _fake_run(queries=["X", "Y"], results=["REJECTED: bad", "No rows. The query was valid"])
    )
    check("counts a rejected query", bad["cypher_errors"] == 1)
    check("half the queries were valid", bad["cypher_valid_rate"] == 0.5)
    check("no rows behind the answer", bad["returned_rows"] is False)

    none_run = metrics.deterministic(_fake_run(queries=[], results=[]))
    check("no cypher at all", none_run["ran_cypher"] is False)
    check("valid rate is undefined, not zero", none_run["cypher_valid_rate"] is None)

    print("\ngold expectations")
    entities = metrics.deterministic(
        _fake_run(), Expectation(entities=["Clonazepam", "Carbamazepine"])
    )
    check("all expected entities present", entities["entities_ok"] is True)
    missing = metrics.deterministic(_fake_run(), Expectation(entities=["Clonazepam", "Aspirin"]))
    check("missing entity is caught", missing["entities_ok"] is False)
    check("reports the ratio", missing["entities_found"] == "1/2")

    number = metrics.deterministic(_fake_run(answer="There are 137 diseases."), Expectation(number=137))
    check("exact number found", number["number_ok"] is True)
    check(
        "substring of a longer number does not count",
        metrics.deterministic(_fake_run(answer="1370 diseases"), Expectation(number=137))[
            "number_ok"
        ]
        is False,
    )
    check(
        "thousands separators are accepted",
        metrics.deterministic(_fake_run(answer="1,552 genes"), Expectation(number=1552))[
            "number_ok"
        ]
        is True,
    )

    declines = [
        "No results were found for that disease.",
        "I couldn't find any compounds matching that.",
        "I could not find an answer to that question.",
        "Hetionet does not contain a node for that disease.",
        "Hetionet does not record who discovered a compound.",
        "The query returned no rows, so the graph has nothing on this.",
        "There is no such disease in the graph.",
        "I was unable to find any matching entries.",
    ]
    for answer in declines:
        got = metrics.deterministic(_fake_run(answer=answer), Expectation(decline=True))
        check(f"declines: {answer[:38]!r}", got["decline_ok"] is True)
    # The other direction matters just as much: a real answer misread as a
    # decline would silently pass a no-answer question the agent got wrong.
    not_declines = [
        "Penicillin was discovered by Alexander Fleming in 1928.",
        "At least 25 compounds treat epilepsy syndrome, including Clonazepam.",
        "The graph contains 137 diseases.",
        "Metformin causes abdominal pain, nausea and diarrhoea.",
    ]
    for answer in not_declines:
        got = metrics.deterministic(_fake_run(answer=answer), Expectation(decline=True))
        check(f"not a decline: {answer[:38]!r}", got["decline_ok"] is False)

    print("\njudge cross-check")
    empty = {"queries": ["X"], "results": ["No rows. The query"]}
    claimed = metrics.deterministic(
        _fake_run(**empty, answer="Epilepsy is treated with Clonazepam.")
    )
    check("flags a high score with no rows", metrics.cross_check(claimed, 5) is True)
    check("low score with no rows is not suspect", metrics.cross_check(claimed, 1) is False)
    check("unjudged is not suspect", metrics.cross_check(claimed, None) is False)
    # A refusal legitimately has no rows behind it, so it must not be flagged.
    refused = metrics.deterministic(_fake_run(**empty, answer="I could not find any results."))
    check("a correct refusal is not suspect", metrics.cross_check(refused, 5) is False)


def test_judge_contract() -> None:
    print("\njudge schema and prompt")
    for category in ("simple", "multi-hop", "aggregate", "ambiguous", "no-answer"):
        check(f"rubric exists: {category}", category in RUBRICS)

    ok = Judgment(reasoning="Fine.", groundedness=5, relevance=4, expectation_met=True)
    check("accepts an in-range judgment", ok.groundedness == 5)
    for bad in (0, 6, 9):
        try:
            Judgment(reasoning="x", groundedness=bad, relevance=3, expectation_met=True)
            check(f"rejects groundedness={bad}", False, "no error raised")
        except Exception:
            check(f"rejects groundedness={bad}", True)

    check("no expectation means no prompt block", _expected_block(None) == "")
    check("an empty expectation adds nothing", _expected_block(Expectation()) == "")
    block = _expected_block(Expectation(entities=["Clonazepam"], number=137))
    check("expectation reaches the prompt", "Clonazepam" in block and "137" in block)

    long_rows = "x" * 9000
    check("truncates oversized rows", len(_truncate(long_rows)) < 9000)
    check("short rows pass through", _truncate("tiny") == "tiny")


def test_observability_optional() -> None:
    """With no keys set the whole module must be inert — the project has to
    run for anyone who never opts into tracing."""
    print("\ntracing is optional")
    if observability.enabled():
        print("  skip  (Langfuse keys are configured in this environment)")
        return
    check("disabled without keys", observability.enabled() is False)
    check("no callbacks", observability.callbacks() == [])
    check("no trace id", observability.new_trace_id() is None)
    check("no metadata", observability.metadata(session_id="x", tags=["y"]) == {})
    check("no trace url", observability.trace_url("abc") is None)
    observability.record_scores("abc", {"groundedness": 5})  # must not raise
    observability.flush()
    check("scoring and flushing are no-ops", True)


def test_summary() -> None:
    print("\nsummary table")
    records = [
        {
            "category": "simple",
            "metrics": metrics.deterministic(_fake_run()),
            "judgment": {
                "groundedness": 5,
                "relevance": 4,
                "expectation_met": True,
                "unsupported_claims": [],
            },
            "suspect": False,
        },
        {
            "category": "no-answer",
            "metrics": metrics.deterministic(_fake_run(queries=[], results=[])),
            "judgment": None,
            "suspect": False,
        },
    ]
    summary = metrics.summarize(records)
    check("counts every record", summary["overall"]["n"] == 2)
    check("averages only judged records", summary["overall"]["groundedness"] == 5.0)
    check("counts the unjudged one", summary["overall"]["unjudged"] == 1)
    check("splits by category", set(summary["by_category"]) == {"simple", "no-answer"})
    table = metrics.format_table(summary)
    check("renders a table with both categories", "simple" in table and "no-answer" in table)
    check("renders missing values as a dash", "-" in table)


def main() -> int:
    test_validate()
    test_strip_reasoning()
    test_trace_callback()
    test_metrics()
    test_judge_contract()
    test_observability_optional()
    test_summary()
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
