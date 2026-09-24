"""Rule-based signals computed from a run, plus the eval summary table.

These cost nothing — they come straight off the AgentRun — and they exist
because an LLM judge on its own is noisy. They also cross-check it: an answer
the judge scored highly despite the agent never getting a single row back is a
judge failure, and `suspect` flags exactly that.

Expectation lives here rather than in evaluate.py so that both the gold checks
and the judge prompt can share one definition without importing the dataset.
"""

import re
from dataclasses import dataclass, field
from statistics import mean

import config
from agent import AgentRun

# Phrases that mark an explicit "the graph has nothing" response: a negation
# followed closely by a finding/holding verb. Only a tripwire — the judge makes
# the real call on whether a refusal was correct.
_DECLINE = re.compile(
    r"\b(?:could\s*n[o']?t|can\s*not|cannot|do(?:es)?\s*n[o']?t|did\s*n[o']?t|is\s*n[o']?t|"
    r"were\s*n[o']?t|was\s*n[o']?t|unable to|without|no|none|nothing|not)\b"
    r"[^.!?]{0,40}?"
    r"\b(?:find|found|contain|include|have|has|record|exist|return(?:ed)?|answer|match(?:es|ing)?|"
    r"results?|rows?|data|information|entr(?:y|ies)|node|such|graph|hetionet|database)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Expectation:
    """A hand-verified fact about what a correct answer contains.

    Only a subset of questions carry one. Reference-free judging is blind to
    the dominant text2cypher failure — a wrong query returning real rows that
    then get faithfully summarised, which scores 5/5 on groundedness — so these
    are the regression tripwire for that case.
    """

    entities: list[str] = field(default_factory=list)  # must all appear in the answer
    number: int | None = None  # exact count the answer must state
    decline: bool = False  # must report that the graph has no answer


def deterministic(run: AgentRun, expected: Expectation | None = None) -> dict:
    """Signals derivable from the run without calling a model."""
    answer = run.answer or ""
    errors = sum(1 for r in run.results if r.startswith(("REJECTED:", "ERROR:")))
    empty = sum(1 for r in run.results if r.startswith("No rows."))

    result = {
        "ran_cypher": bool(run.queries),
        "query_count": run.query_count,
        "hit_query_budget": run.query_count >= config.MAX_QUERIES,
        "cypher_errors": errors,
        "cypher_valid_rate": round(1 - errors / run.query_count, 2) if run.queries else None,
        "returned_rows": run.query_count > errors + empty,
        # Always computed, not just on annotated questions: cross_check needs it
        # to tell a legitimate refusal apart from an unevidenced claim.
        "declined": bool(_DECLINE.search(answer)) or run.graceful_failure,
        "graceful_failure": run.graceful_failure,
        "crashed": run.error is not None,
        "latency_s": run.latency_s,
        "tokens": run.tokens,
    }

    if expected:
        if expected.entities:
            found = [e for e in expected.entities if e.lower() in answer.lower()]
            result["entities_found"] = f"{len(found)}/{len(expected.entities)}"
            result["entities_ok"] = len(found) == len(expected.entities)
        if expected.number is not None:
            result["number_ok"] = _states_number(answer, expected.number)
        if expected.decline:
            result["decline_ok"] = result["declined"]

    return result


def cross_check(signals: dict, groundedness: int | None) -> bool:
    """True when the judge liked an answer that had no evidence to stand on.

    An answer that declined is excluded: having no rows is the correct outcome
    there, not a missing citation.
    """
    if groundedness is None:
        return False
    return (
        groundedness >= config.JUDGE_PASS_SCORE
        and not signals["returned_rows"]
        and not signals["declined"]
    )


def _states_number(answer: str, expected: int) -> bool:
    """Whether the answer states the number, with or without thousands separators."""
    digits = {str(expected), f"{expected:,}"}
    return any(re.search(rf"(?<![\d,]){re.escape(d)}(?![\d,])", answer) for d in digits)


# --- summary -------------------------------------------------------------
PASS_FIELDS = ("entities_ok", "number_ok", "decline_ok")

# Signals worth charting in Langfuse. Tokens and latency are omitted: the trace
# already carries both natively, and duplicating them as scores would double-count
# in any dashboard built on either one.
SCORE_FIELDS = (
    "cypher_valid_rate",
    "cypher_errors",
    "query_count",
    "ran_cypher",
    "returned_rows",
    "hit_query_budget",
    "graceful_failure",
    "declined",
    "crashed",
    *PASS_FIELDS,
)


def summarize(records: list[dict]) -> dict:
    """Per-category and overall aggregates over the finished records."""
    by_category: dict[str, dict] = {}
    for category in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == category]
        by_category[category] = _aggregate(rows)
    return {"overall": _aggregate(records), "by_category": by_category}


def _aggregate(records: list[dict]) -> dict:
    judged = [r["judgment"] for r in records if r.get("judgment")]
    signals = [r["metrics"] for r in records if r.get("metrics")]
    gold = [s[f] for s in signals for f in PASS_FIELDS if f in s]

    return {
        "n": len(records),
        "groundedness": _mean(j["groundedness"] for j in judged),
        "relevance": _mean(j["relevance"] for j in judged),
        "expectation_met": _rate(j["expectation_met"] for j in judged),
        "gold_checks": _rate(gold) if gold else None,
        "cypher_valid": _mean(
            s["cypher_valid_rate"] for s in signals if s["cypher_valid_rate"] is not None
        ),
        "avg_queries": _mean(s["query_count"] for s in signals),
        "avg_latency_s": _mean(s["latency_s"] for s in signals),
        "tokens": sum(s["tokens"] for s in signals),
        "crashed": sum(1 for s in signals if s["crashed"]),
        "suspect": sum(1 for r in records if r.get("suspect")),
        "unjudged": len(records) - len(judged),
    }


def _mean(values) -> float | None:
    values = list(values)
    return round(mean(values), 2) if values else None


def _rate(values) -> float | None:
    values = list(values)
    return round(sum(1 for v in values if v) / len(values), 2) if values else None


_HEADER = "{:<12} {:>3} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>8}"
_COLUMNS = ("category", "n", "ground", "relev", "expect", "gold", "cypher", "qrys", "tokens")


def format_table(summary: dict) -> str:
    """The eval summary, in evaluate.py's existing plain-text style."""
    lines = [_HEADER.format(*_COLUMNS), "-" * 74]
    for category, stats in summary["by_category"].items():
        lines.append(_row(category, stats))
    lines += ["-" * 74, _row("OVERALL", summary["overall"])]

    overall = summary["overall"]
    notes = [
        f"{overall['crashed']} crashed" if overall["crashed"] else "",
        f"{overall['unjudged']} unjudged" if overall["unjudged"] else "",
        f"{overall['suspect']} suspect (judge passed an answer with no rows)"
        if overall["suspect"]
        else "",
        f"avg {overall['avg_latency_s']}s/question" if overall["avg_latency_s"] else "",
    ]
    lines.append("")
    lines.append("  ".join(n for n in notes if n))
    return "\n".join(lines)


def _row(label: str, stats: dict) -> str:
    return _HEADER.format(
        label[:12],
        stats["n"],
        _fmt(stats["groundedness"]),
        _fmt(stats["relevance"]),
        _fmt(stats["expectation_met"]),
        _fmt(stats["gold_checks"]),
        _fmt(stats["cypher_valid"]),
        _fmt(stats["avg_queries"]),
        stats["tokens"],
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"
