"""Plain-English question -> Cypher -> Neo4j -> plain-English answer.

A LangGraph ReAct agent with one tool, ``run_cypher``. The model writes a query,
sees the rows, and decides for itself whether to answer or query again — so it
can look a name up first, then use it, or split a comparison into two queries.

    question -> LLM -+-> run_cypher -> rows -+-> LLM -> ... -> answer
                     ^                       |
                     +-----------------------+

Everything is assembled from framework primitives: a ChatPromptTemplate holds the
system prompt, create_react_agent owns the loop, an InMemorySaver checkpointer
carries conversation history, and the answer is extracted by an LCEL pipe whose
.with_fallbacks() turns a runaway loop into a polite message.

Tool failures are returned to the model as text rather than raised, because a
rejected or empty query is something it can recover from on the next turn.
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

import config
import observability
from examples import format_examples
from graph import HetionetGraph
from validate import UnsafeQueryError, validate

log = logging.getLogger("hetionet")

GRACEFUL_FAILURE = (
    "I couldn't find a reliable answer to that. Try rephrasing the question, or "
    "naming the disease, compound or gene the way Hetionet would (for example "
    '"myocardial infarction" rather than "heart attack").'
)

SYSTEM = """You answer questions about the Hetionet biomedical knowledge graph by querying it with Cypher.

You have one tool, run_cypher. Call it to run a read-only Cypher query and get the rows back.
You may call it more than once: if a query returns nothing, or you need one fact before you can
look up another, query again. When you have enough data, answer in plain English.

Graph schema:
{schema}

Writing the Cypher:
- Read-only only. Never CREATE, MERGE, DELETE, SET, REMOVE, DROP or LOAD CSV.
- Use only the labels, relationship types and properties in the schema above. Do not invent any.
- Relationship types are abbreviated (TREATS_CtD, ASSOCIATES_DaG, ...). Use them exactly as written.
- Every relationship must appear in the metagraph above with that exact direction AND those exact
  endpoint labels. CAUSES_CcSE only goes (:Compound)->(:SideEffect), never from a Disease. When two
  hops share a node, use separate MATCH clauses reusing the same variable rather than chaining
  through a node that does not support both.
- The target is Neo4j 3.5, so CALL {{ ... }} subqueries, EXISTS {{ ... }} and newer aggregation
  syntax are NOT available. Chain multiple counts with WITH instead. The only procedures you may
  CALL are db.labels() and db.relationshipTypes().
- Node names come from formal biomedical ontologies (Disease Ontology, DrugBank, Uberon), not
  everyday speech. Translate colloquial terms first: "heart attack" -> "myocardial infarction",
  "high blood pressure" -> "hypertension", "sugar diabetes" -> "diabetes mellitus".
- Match entity names case-insensitively with toLower(x.name) CONTAINS toLower('...').
- End with LIMIT 25 unless the user asks for a specific number or the query returns a single
  aggregate such as count().
- Return named columns (RETURN g.name AS gene), not whole nodes.

If a query returns no rows, do not assume the answer is "nothing". First check whether the entity
exists under another name — for example MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'heart'
RETURN d.name — and retry with what you find. Only report that the graph has no answer once you
have actually looked.

Answering:
- Use ONLY data returned by the tool. Never add biomedical facts from your own knowledge, and
  never guess.
- Results are capped at {max_rows} rows, so do not state a total unless a count() column gave you
  one. Say "at least N" or "including" instead.
- Be concise. For a long list, name a few examples.
- Earlier turns in this conversation are visible to you, so resolve follow-up references like
  "that drug" or "the first one" against what you already reported.

Worked examples of the Cypher (not of the tool call):
{examples}"""


def _strip_reasoning(text: str) -> str:
    """Drop <think> blocks that reasoning models (Qwen, gpt-oss) emit.

    No framework primitive does this — LangChain's output parsers assume the
    content is already clean — so it stays a plain function, piped in as a
    RunnableLambda below.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # An unterminated <think> means the response was cut off mid-reasoning, so
    # there is no clean answer boundary. The last paragraph is the best guess.
    if re.search(r"<think>", text, re.IGNORECASE):
        tail = re.split(r"<think>", text, flags=re.IGNORECASE)[-1]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", tail) if p.strip()]
        text = paragraphs[-1] if paragraphs else tail

    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def _make_run_cypher(graph: HetionetGraph):
    """The agent's only tool, closed over the graph connection."""

    @tool
    def run_cypher(cypher: str) -> str:
        """Run a read-only Cypher query against Hetionet and return the rows as JSON.

        Args:
            cypher: A single read-only Cypher query.
        """
        log.info("Cypher:\n%s", cypher)
        # Failures come back as text, not exceptions: the model is expected to
        # read them and fix its own query on the next turn.
        try:
            validate(cypher)
        except UnsafeQueryError as exc:
            log.warning("Rejected: %s", exc)
            return f"REJECTED: {exc}"

        try:
            rows = graph.query(cypher)
        except Exception as exc:  # noqa: BLE001 - handed to the model verbatim
            log.warning("Failed: %s", exc)
            return f"ERROR: {type(exc).__name__}: {exc}. Fix the query and try again."

        log.info("-> %d row(s)", len(rows))
        if not rows:
            return (
                "No rows. The query was valid, so either a name literal does not match "
                "Hetionet's formal ontology naming, or a relationship hop is not in the "
                "metagraph. Search for the entity's real name before concluding the graph "
                "has no answer."
            )
        note = (
            f"\n(Truncated at the {config.MAX_ROWS}-row cap; the real total is higher.)"
            if len(rows) >= config.MAX_ROWS
            else ""
        )
        return json.dumps(rows, indent=2) + note

    return run_cypher


def build_prompt(graph: HetionetGraph) -> ChatPromptTemplate:
    """System prompt + the running conversation.

    MessagesPlaceholder is filled from the agent's own state, so the same
    template serves the first turn and every tool round-trip after it. The
    schema and examples are partial()'d in once at startup — they never change.
    """
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM), MessagesPlaceholder("messages")]
    ).partial(
        schema=graph.schema,
        examples=format_examples(),
        max_rows=str(config.MAX_ROWS),
    )


def build_agent(
    graph: HetionetGraph,
    model: str = config.GROQ_MODEL,
    remember: bool = True,
) -> Runnable:
    """Compile the agent. Build once, then pass it to ask() per question."""
    # max_retries covers Groq's 429s, which the free tier hits easily.
    llm = ChatGroq(
        model=model,
        temperature=0,
        api_key=config.groq_api_key(),
        max_retries=config.LLM_MAX_RETRIES,
    )

    agent = create_react_agent(
        llm,
        [_make_run_cypher(graph)],
        prompt=build_prompt(graph),
        # Persists state per thread_id, which is what lets a follow-up question
        # refer back to an earlier answer.
        checkpointer=InMemorySaver() if remember else None,
    )

    # An LCEL chain of question -> agent state -> final message -> clean text,
    # so the whole thing is one str -> str Runnable you can invoke, stream or
    # batch like any other chain.
    chain = (
        RunnableLambda(lambda question: {"messages": [("user", question)]})
        # Each query costs two steps (model + tool), plus the final answer.
        | agent.with_config(recursion_limit=2 * config.MAX_QUERIES + 1)
        | RunnableLambda(lambda state: state["messages"][-1])
        | StrOutputParser()
        | RunnableLambda(_strip_reasoning)
    )

    # Overrunning the loop is the one failure the model cannot talk its way out
    # of, so it degrades to a fixed message instead of raising.
    return chain.with_fallbacks(
        [RunnableLambda(lambda _: GRACEFUL_FAILURE)],
        exceptions_to_handle=(GraphRecursionError,),
    ).with_config(run_name="hetionet_text2cypher")


def ask(agent: Runnable, question: str, thread_id: str = "default") -> str:
    """Run one question through the chain and return the final answer.

    Questions sharing a thread_id share history; a new thread_id starts fresh.
    """
    return agent.invoke(question, _run_config(thread_id))


def _run_config(
    thread_id: str,
    trace_id: str | None = None,
    extra_callbacks: list | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Thread id, Langfuse callbacks and trace metadata in one runnable config.

    With tracing off, callbacks and metadata are empty and this is exactly the
    config the agent has always been invoked with.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": [*observability.callbacks(trace_id), *(extra_callbacks or [])],
        "metadata": observability.metadata(session_id=thread_id, tags=tags),
    }


# --- Traced runs, for evaluation -----------------------------------------
@dataclass(slots=True)
class AgentRun:
    """Everything one question produced, not just its final answer.

    An LLM judge cannot score groundedness from the answer alone — it needs the
    Cypher the agent ran and the rows that came back, which the LCEL chain
    discards when it takes ``state["messages"][-1]``.
    """

    question: str
    answer: str = ""
    queries: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)  # index-aligned with queries
    tokens: int = 0
    latency_s: float = 0.0
    graceful_failure: bool = False
    trace_id: str | None = None
    error: str | None = None

    @property
    def query_count(self) -> int:
        return len(self.queries)

    def to_dict(self) -> dict:
        return asdict(self) | {"query_count": self.query_count}


class _CypherTrace(BaseCallbackHandler):
    """Records tool calls and token usage as the graph runs.

    A callback rather than a read of the final state, because it fires *during*
    the run: when the agent overruns MAX_QUERIES the GraphRecursionError
    destroys the state before the fallback returns GRACEFUL_FAILURE, and the
    overrun is precisely the case worth inspecting.
    """

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.results: list[str] = []
        self.tokens = 0

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        inputs = kwargs.get("inputs") or {}
        self.queries.append(inputs.get("cypher") or input_str or "")

    def on_tool_end(self, output, **kwargs) -> None:
        self.results.append(str(getattr(output, "content", output)))

    def on_llm_end(self, response, **kwargs) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.tokens += usage.get("total_tokens") or 0


def ask_traced(agent: Runnable, question: str, thread_id: str = "default") -> AgentRun:
    """Run one question and return the full record of what happened.

    Invokes the same Runnable ``ask()`` does, so the thing being evaluated is
    the thing that ships.
    """
    trace = _CypherTrace()
    trace_id = observability.new_trace_id()
    run = AgentRun(question=question, trace_id=trace_id)
    started = time.perf_counter()
    try:
        run.answer = agent.invoke(
            question,
            _run_config(thread_id, trace_id, extra_callbacks=[trace], tags=["eval"]),
        )
    except Exception as exc:  # noqa: BLE001 - recorded so the eval keeps going
        run.error = f"{type(exc).__name__}: {exc}"
    run.latency_s = round(time.perf_counter() - started, 2)

    # Read off the callback, so a run that crashed still reports its queries.
    run.queries, run.results, run.tokens = trace.queries, trace.results, trace.tokens
    run.graceful_failure = run.answer == GRACEFUL_FAILURE
    return run
