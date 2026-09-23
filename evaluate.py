"""Run the question set through the agent and score every answer.

Usage: python evaluate.py [--only CATEGORY] [--no-judge] [--limit N]

Each question produces three kinds of evidence:

  deterministic  rule-based signals off the run itself — did it run any Cypher,
                 how many queries, were any rejected (metrics.py)
  gold           hand-verified expectations, on the subset of questions where
                 the right answer is stable (golden_dataset/)
  judge          an LLM scoring groundedness and relevance against the rows the
                 agent actually saw (judge.py)

Judging roughly doubles token spend, so --no-judge skips it and --judge-only
re-scores a saved results file without calling the agent at all, which is how
you iterate on the rubric cheaply.

When Langfuse is configured, each run is traced and its scores are attached to
that trace, so the summary below and the timeline in the UI are the same data.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import config
import metrics
import observability
from agent import AgentRun, ask_traced, build_agent
from graph import HetionetGraph
from judge import build_judge, judge_run
from metrics import Expectation


GOLDEN_DATASET = Path(__file__).parent / "golden_dataset" / "questions.json"


@dataclass(slots=True)
class EvalCase:
    category: str
    question: str
    expected: Expectation | None = None
    reference_answer: str | None = None


def load_cases(path: Path = GOLDEN_DATASET) -> list[EvalCase]:
    """Read the golden dataset.

    The question set lives as data rather than Python literals so the graph
    values it asserts can be diffed and re-verified independently of the runner.
    See golden_dataset/README.md for how each entry was obtained.
    """
    raw = json.loads(path.read_text())["questions"]
    return [
        EvalCase(
            category=entry["category"],
            question=entry["question"],
            expected=Expectation(**entry["expected"]) if entry.get("expected") else None,
            reference_answer=entry.get("reference_answer"),
        )
        for entry in raw
    ]


EVAL_QUESTIONS: list[EvalCase] = load_cases()


def score(record: dict, run: AgentRun, case: EvalCase, judge) -> dict:
    """Attach deterministic signals, the judgment and Langfuse scores to a record."""
    signals = metrics.deterministic(run, case.expected)
    judgment = (
        judge_run(judge, run, case.category, case.expected, case.reference_answer)
        if judge
        else None
    )

    record |= {
        "metrics": signals,
        "judgment": judgment.model_dump() if judgment else None,
        "suspect": metrics.cross_check(
            signals, judgment.groundedness if judgment else None
        ),
    }

    if judgment:
        observability.record_scores(
            run.trace_id,
            {
                "groundedness": judgment.groundedness,
                "relevance": judgment.relevance,
                "expectation_met": judgment.expectation_met,
                # Carried as its own score rather than only as a comment:
                # Langfuse Cloud drops score comments on the current ingestion
                # path, so the reasoning would otherwise be invisible in the UI.
                "judge_reasoning": judgment.reasoning[:480],
            },
            comment=judgment.reasoning,
        )
    # Rule-based signals are recorded even with --no-judge: they cost nothing and
    # they are what makes Cypher validity and the judge watchdog chartable.
    observability.record_scores(
        run.trace_id,
        {k: v for k, v in signals.items() if k in metrics.SCORE_FIELDS}
        | {"suspect": record["suspect"]},
    )
    return record


def run_eval(cases: list[EvalCase], judge, delay: float) -> list[dict]:
    records = []
    with HetionetGraph() as graph:
        # Each eval question is independent: no history should leak between them.
        agent = build_agent(graph, remember=False)
        for index, case in enumerate(cases, start=1):
            if index > 1 and delay:
                time.sleep(delay)
            print(f"\n[{index}/{len(cases)}] ({case.category}) {case.question}")

            run = ask_traced(agent, case.question)
            record = {"category": case.category, **run.to_dict()}
            try:
                record = score(record, run, case, judge)
            except Exception as exc:  # noqa: BLE001 - a bad judgment is not a failed run
                record |= {"judge_error": f"{type(exc).__name__}: {exc}"}

            records.append(record)
            _print_record(record)
    return records


def rejudge(records: list[dict], judge) -> list[dict]:
    """Re-score saved records without calling the agent, for tuning the rubric."""
    by_question = {case.question: case for case in EVAL_QUESTIONS}
    rescored = []
    for index, record in enumerate(records, start=1):
        case = by_question.get(record["question"]) or EvalCase(
            record["category"], record["question"]
        )
        run = AgentRun(
            question=record["question"],
            answer=record.get("answer", ""),
            queries=record.get("queries", []),
            results=record.get("results", []),
            tokens=record.get("tokens", 0),
            latency_s=record.get("latency_s", 0.0),
            graceful_failure=record.get("graceful_failure", False),
            trace_id=record.get("trace_id"),
            error=record.get("error"),
        )
        print(f"\n[{index}/{len(records)}] ({case.category}) {case.question}")
        rescored.append(score(dict(record), run, case, judge))
        _print_record(rescored[-1])
    return rescored


def _print_record(record: dict) -> None:
    answer = record.get("answer") or record.get("error") or ""
    print(f"  answer   : {answer[:160]}")
    if judgment := record.get("judgment"):
        print(
            f"  judged   : grounded {judgment['groundedness']}/5  "
            f"relevant {judgment['relevance']}/5  "
            f"expected {'yes' if judgment['expectation_met'] else 'NO'}"
        )
        if claims := judgment["unsupported_claims"]:
            print(f"  UNSUPPORTED: {', '.join(claims[:4])}")
    if record.get("suspect"):
        print("  SUSPECT  : judge passed an answer that had no rows behind it")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to pause between questions, to stay under Groq rate limits.",
    )
    parser.add_argument(
        "--only",
        help="Run just one category (simple, multi-hop, aggregate, ambiguous, no-answer). "
        "A full run costs roughly 45k Groq tokens, so partial runs matter on the free tier.",
    )
    parser.add_argument("--limit", type=int, help="Stop after N questions.")
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the LLM judge; keep the rule-based metrics."
    )
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL)
    parser.add_argument(
        "--judge-only",
        metavar="FILE",
        help="Re-score a saved results file. Calls the judge but never the agent, "
        "which is the cheap way to iterate on the rubric.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    cases = [c for c in EVAL_QUESTIONS if not args.only or c.category == args.only]
    if not cases:
        parser.error(f"no questions in category {args.only!r}")
    if args.limit:
        cases = cases[: args.limit]

    judge = None if args.no_judge else build_judge(args.judge_model)

    if args.judge_only:
        with open(args.judge_only) as fh:
            saved = json.load(fh)
        records = rejudge(saved.get("records", saved)[: args.limit], judge)
    else:
        records = run_eval(cases, judge, args.delay)

    summary = metrics.summarize(records)
    payload = {
        "meta": {
            "agent_model": config.GROQ_MODEL,
            "judge_model": None if args.no_judge else args.judge_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": summary,
        "records": records,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    observability.flush()

    print(f"\n{'=' * 74}")
    print(metrics.format_table(summary))
    print(f"\nwritten to {args.out}")
    if observability.enabled():
        print(f"traces: {config.LANGFUSE_BASE_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
