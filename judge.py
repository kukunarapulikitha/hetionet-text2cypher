"""LLM-as-a-judge scoring for one agent run.

The agent's system prompt makes a checkable promise — "Use ONLY data returned
by the tool. Never add biomedical facts from your own knowledge" — and nothing
verified it. This does: the judge sees the Cypher, the rows those queries
actually returned, and the answer, and rates whether the answer is supported.

One judge with a per-category rubric rather than one judge per criterion: the
criteria are highly correlated, and three calls per answer would triple the
token spend for very little extra signal.

Structured output comes from ``with_structured_output(include_raw=True)``, so a
model that returns malformed JSON surfaces a parsing error instead of raising —
a bad judge response should cost one score, not the whole eval run.
"""

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

import config
from agent import AgentRun
from metrics import Expectation

log = logging.getLogger("hetionet")


class Judgment(BaseModel):
    """One graded answer."""

    # Declared first so the model writes its reasoning before committing to
    # scores, rather than rationalising a number it already emitted.
    reasoning: str = Field(description="Two sentences at most, justifying the scores.")
    groundedness: int = Field(ge=1, le=5, description="Is every claim supported by the rows?")
    relevance: int = Field(ge=1, le=5, description="Does it answer the question asked?")
    expectation_met: bool = Field(description="Does it satisfy the category rubric?")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims absent from the returned rows."
    )


RUBRICS: dict[str, str] = {
    "simple": "Should answer the single relationship asked about, naming entities from the rows.",
    "multi-hop": (
        "Needs facts chained across two or more relationships. Should name entities from the "
        "rows, and not silently answer an easier one-hop version of the question."
    ),
    "aggregate": (
        "Should report a number that came from a count() column. A count of the rows listed is "
        "NOT acceptable, because results are capped at "
        f"{config.MAX_ROWS} rows."
    ),
    "ambiguous": (
        "The question uses a colloquial term the graph does not use (heart attack, sugar "
        "diabetes, Lou Gehrig's disease). The agent should have resolved it to the formal "
        "ontology name and answered. Failing to translate the term is a failure."
    ),
    "no-answer": (
        "The graph genuinely cannot answer this. A correct response explicitly says so, or says "
        "it found nothing. It must name ZERO specific biomedical entities as an answer. Supplying "
        "a real-world answer from the model's own knowledge is the WORST case here and must score "
        "groundedness 1 — even if that answer is factually true."
    ),
}

JUDGE_SYSTEM = """You grade a text-to-Cypher agent that answers questions from the Hetionet \
biomedical knowledge graph. You are strict and you do not reward fluency.

The agent is under a hard constraint: it may use ONLY data returned by its Cypher queries. \
Any biomedical fact in the answer that is absent from the returned rows is a violation, even \
if that fact is true in the real world. That is the single most important thing you check.

Scoring groundedness (1-5):
  5 - every claim traces to a returned row.
  3 - mostly supported, with a detail that does not appear in the rows.
  1 - names entities or facts that never appeared in any row, or answers from world knowledge.
An answer that asserts nothing — it reports that the graph has no answer, or that it found
nothing — is fully grounded and scores 5. There is nothing in it left unsupported. Saying
"I found no results" is NOT an unsupported claim, and statements about what the graph does
or does not contain are not biomedical claims. Only ever penalise a *positive* assertion
about biology that the rows do not back up.
Results are capped at {max_rows} rows, so an answer that states a definite total when the rows
hit that cap is overstating its evidence unless a count() column supplied the number, or it
hedged with "at least" or "including".

Scoring relevance (1-5): does it answer the question that was actually asked?

Category rubric for this question ({category}):
{rubric}

Set expectation_met to whether that rubric is satisfied. List every unsupported claim you find."""

JUDGE_HUMAN = """Question:
{question}

Cypher the agent ran:
{queries}

Rows the queries returned:
{results}

{expected}The agent's answer:
{answer}"""


def build_judge(model: str | None = None) -> Runnable:
    """A question+run -> Judgment chain. Build once, reuse for every answer."""
    model = model or config.JUDGE_MODEL
    if model == config.GROQ_MODEL:
        log.warning(
            "JUDGE_MODEL is the same model as the agent (%s); a model grading its own "
            "output scores it too generously. Set JUDGE_MODEL to something else.",
            model,
        )

    llm = ChatGroq(
        model=model,
        temperature=0,
        api_key=config.groq_api_key(),
        max_retries=config.LLM_MAX_RETRIES,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)]
    ).partial(max_rows=str(config.MAX_ROWS))

    chain = (
        prompt
        | llm.with_structured_output(Judgment, include_raw=True)
        | RunnableLambda(_unwrap)
    )
    # A judge that fails is a missing score, never a failed eval run.
    return chain.with_retry(
        stop_after_attempt=config.JUDGE_MAX_RETRIES
    ).with_fallbacks([RunnableLambda(lambda _: None)]).with_config(run_name="judge")


def _unwrap(raw: dict) -> Judgment:
    """Turn include_raw's envelope back into a Judgment, or raise to trigger a retry."""
    if raw.get("parsing_error") or not raw.get("parsed"):
        raise ValueError(f"judge returned unparseable output: {raw.get('parsing_error')}")
    return raw["parsed"]


def judge_run(
    judge: Runnable,
    run: AgentRun,
    category: str,
    expected: Expectation | None = None,
) -> Judgment | None:
    """Score one run. Returns None if the judge could not be parsed."""
    return judge.invoke(
        {
            "question": run.question,
            "category": category,
            "rubric": RUBRICS.get(category, "Answer the question using only the returned rows."),
            "queries": _numbered(run.queries) or "(the agent ran no queries at all)",
            "results": _truncate(_numbered(run.results)) or "(no rows were returned)",
            "answer": run.answer or "(no answer produced)",
            "expected": _expected_block(expected),
        }
    )


def _numbered(items: list[str]) -> str:
    return "\n".join(f"[{i}] {item}" for i, item in enumerate(items, start=1))


def _truncate(text: str) -> str:
    limit = config.JUDGE_MAX_RESULT_CHARS
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... (truncated, {len(text) - limit} more characters)"


def _expected_block(expected: Expectation | None) -> str:
    """Fold the gold annotation into the prompt, when a question has one.

    Groundedness alone cannot catch the common text2cypher failure of a wrong
    query returning real rows that get faithfully summarised, so on annotated
    questions the judge is told what a right answer contains.
    """
    if expected is None:
        return ""
    known = {
        field: value
        for field in ("entities", "number", "decline")
        if (value := getattr(expected, field))
    }
    if not known:
        return ""
    return (
        "A correct answer is known to satisfy the following. Treat a conflict with this as a "
        f"failure of expectation_met:\n{json.dumps(known, indent=2)}\n\n"
    )
