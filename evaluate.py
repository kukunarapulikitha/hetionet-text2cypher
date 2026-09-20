"""Run a fixed question set through the agent and log what happened.

Usage: python evaluate.py [--out eval_results.json]

Categories: simple 1-hop, multi-hop, aggregate, ambiguous entity naming, and
questions with no valid answer in the graph.
"""

import argparse
import json
import logging
import time

from agent import ask, build_agent
from graph import HetionetGraph

EVAL_QUESTIONS: list[tuple[str, str]] = [
    # -- simple 1-hop ----------------------------------------------------
    ("simple", "What compounds treat epilepsy syndrome?"),
    ("simple", "Which genes are associated with Crohn's disease?"),
    ("simple", "What symptoms does asthma present?"),
    ("simple", "What side effects does Metformin cause?"),
    ("simple", "Which anatomies express the gene BRCA1?"),
    # -- multi-hop -------------------------------------------------------
    ("multi-hop", "Which compounds bind genes associated with multiple sclerosis?"),
    ("multi-hop", "What pathways do genes associated with type 2 diabetes participate in?"),
    ("multi-hop", "Which diseases share genes with breast cancer?"),
    ("multi-hop", "What side effects are caused by compounds that treat hypertension?"),
    ("multi-hop", "Which pharmacologic classes include compounds that treat asthma?"),
    # -- aggregates ------------------------------------------------------
    ("aggregate", "How many diseases are in the graph?"),
    ("aggregate", "Which 10 compounds treat the most diseases?"),
    ("aggregate", "What are the five most common side effects across all compounds?"),
    # -- ambiguous entity naming ----------------------------------------
    # The graph name differs from everyday usage, but the entity IS present:
    # high blood pressure -> hypertension, sugar diabetes -> diabetes mellitus,
    # Lou Gehrig's disease -> amyotrophic lateral sclerosis.
    ("ambiguous", "What treats high blood pressure?"),
    ("ambiguous", "Which genes are linked to sugar diabetes?"),
    ("ambiguous", "What genes are associated with Lou Gehrig's disease?"),
    # -- expected to have no answer in Hetionet -------------------------
    # "heart attack" belongs here, not under ambiguous: Hetionet has no
    # myocardial infarction node at all, only coronary artery disease. The
    # right behaviour is to translate the term and then report no results.
    ("no-answer", "What drugs treat heart attack?"),
    ("no-answer", "What compounds treat Klingon lung fever?"),
    ("no-answer", "Which genes are associated with unicorn deficiency syndrome?"),
    ("no-answer", "What is the average cost of insulin in the United States?"),
    ("no-answer", "Who discovered penicillin?"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    questions = [q for q in EVAL_QUESTIONS if not args.only or q[0] == args.only]
    if not questions:
        parser.error(f"no questions in category {args.only!r}")

    results = []
    with HetionetGraph() as graph:
        # Each eval question is independent: no history should leak between them.
        agent = build_agent(graph, remember=False)
        for index, (category, question) in enumerate(questions, start=1):
            if index > 1 and args.delay:
                time.sleep(args.delay)
            print(f"\n[{index}/{len(questions)}] ({category}) {question}")
            try:
                record = {
                    "category": category,
                    "question": question,
                    "answer": ask(agent, question),
                }
            except Exception as exc:  # noqa: BLE001 - record and keep going
                record = {
                    "category": category,
                    "question": question,
                    "crashed": f"{type(exc).__name__}: {exc}",
                }
            results.append(record)
            print(f"  answer   : {(record.get('answer') or record.get('crashed', ''))[:180]}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    crashed = sum(1 for r in results if r.get("crashed"))
    print(f"\n{'=' * 60}")
    print(f"total: {len(results)}   crashed: {crashed}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
