"""CLI entry point.

The core logic lives in agent.py / graph.py, so a FastAPI or Flask wrapper can
import build_agent() and ask() without touching anything here.
"""

import argparse
import logging
import sys

from agent import ask, build_agent
from graph import HetionetGraph

BANNER = """Hetionet plain-English query agent
Ask a biomedical question. Type 'exit' or Ctrl-D to quit.
"""


def configure_logging(show_cypher: bool) -> None:
    # The generated Cypher is logged at INFO for debugging; -q hides it.
    logging.basicConfig(
        level=logging.INFO if show_cypher else logging.WARNING,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    # httpx logs every Groq call at INFO, which drowns out the Cypher.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query Hetionet in plain English.")
    parser.add_argument("question", nargs="*", help="Ask one question and exit.")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Hide the generated Cypher."
    )
    parser.add_argument(
        "--refresh-schema",
        action="store_true",
        help="Re-introspect the graph schema and update the cache.",
    )
    args = parser.parse_args()

    configure_logging(show_cypher=not args.quiet)

    with HetionetGraph() as graph:
        if args.refresh_schema:
            graph.refresh_schema()
            print("Schema cache refreshed.", file=sys.stderr)

        agent = build_agent(graph)

        if args.question:
            print(ask(agent, " ".join(args.question)))
            return 0

        print(BANNER, file=sys.stderr)
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return 0
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                return 0
            try:
                print(f"\n{ask(agent, question)}\n")
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                print(f"\nSomething went wrong: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
