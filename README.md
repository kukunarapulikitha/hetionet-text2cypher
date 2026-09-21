# Hetionet plain-English query agent

Ask a biomedical question in English. The agent reads the Neo4j schema, writes the
Cypher, runs it against [Hetionet](https://het.io/), and answers in plain English.
No Cypher knowledge required.

Built with **LangGraph** + **ChatGroq**, traced with **Langfuse** and scored by an
**LLM judge**. Read-only by construction.

```
$ python main.py "What compounds treat epilepsy syndrome?"
[INFO] Cypher:
MATCH (c:Compound)-[:TREATS_CtD]->(d:Disease)
WHERE toLower(d.name) CONTAINS toLower('epilepsy syndrome')
RETURN c.name AS compound
LIMIT 25
[INFO] -> 25 row(s)

At least 25 compounds treat epilepsy syndrome, including Clonazepam,
Lacosamide, Oxcarbazepine, Propofol and Felbamate.
```

## Setup

```bash
cd hetionet-text2cypher
uv venv --python 3.12
uv pip install -r requirements.txt
cp .env.example .env      # then add your GROQ_API_KEY
```

Get a free Groq key at [console.groq.com/keys](https://console.groq.com/keys).

The Neo4j defaults already point at the **public read-only Hetionet instance**
(`bolt://neo4j.het.io:7687`, auth disabled), so there is nothing to install or load.

## Usage

```bash
python main.py                                  # interactive REPL
python main.py "Which genes associate with Crohn's disease?"   # one-shot
python main.py -q "How many diseases are there?"               # hide the Cypher
python main.py --refresh-schema                 # re-introspect after a graph change
python test_offline.py                          # 60 assertions, no Groq key needed
python evaluate.py                              # run + score the 21-question eval set
python evaluate.py --only simple --no-judge     # cheap subset, rule-based metrics only
```

## How it works

```
question ──> LLM ──┬──> run_cypher ──> rows ──┬──> LLM ──> ... ──> answer
                   ^                          │
                   └──────────────────────────┘
```

A LangGraph ReAct agent (`create_react_agent`) with a single tool, `run_cypher`.
The model writes a query, sees the rows, and decides for itself whether it can
answer or needs to query again — so it can look a name up first and then use it,
or split a comparison into two queries. Simple questions still cost exactly one
query; the loop only engages when the first answer isn't sufficient.

It is assembled entirely from framework primitives — no hand-rolled control flow:

| Piece | Primitive |
| --- | --- |
| System prompt | `ChatPromptTemplate` + `MessagesPlaceholder`, with the schema `.partial()`'d in once at startup |
| Few-shot examples | `FewShotPromptTemplate` (`examples.py`) |
| The loop | `create_react_agent` |
| Conversation memory | `InMemorySaver` checkpointer, keyed by `thread_id` |
| Query budget | `.with_config(recursion_limit=...)`, from `MAX_QUERIES` |
| Graceful overrun | `.with_fallbacks(exceptions_to_handle=(GraphRecursionError,))` |
| `run_cypher` tool | `@tool` → `validate()` → `graph.query()`. 10s timeout, 25-row cap. |

The whole thing composes into one `str -> str` LCEL chain:

```python
chain = (
    RunnableLambda(lambda q: {"messages": [("user", q)]})
    | agent.with_config(recursion_limit=2 * config.MAX_QUERIES + 1)
    | RunnableLambda(lambda state: state["messages"][-1])
    | StrOutputParser()
    | RunnableLambda(_strip_reasoning)
)
```

so `ask()` is a one-liner and the chain can be `.invoke()`d, `.stream()`d or
`.batch()`ed like any other Runnable.

**Follow-up questions work**, because the checkpointer persists state per thread:

```
> What compounds treat epilepsy syndrome? Name just two.
  Clonazepam and Lacosamide.
> What side effects does the first one cause?
  Clonazepam is associated with ataxia, dysphoria, cardiac arrest, ...
```

`ask(agent, q, thread_id="...")` picks the conversation; a new id starts fresh.
`evaluate.py` passes `remember=False` so eval questions can't contaminate each other.

**Failures are returned to the model as text, not raised.** A rejected query, a
Cypher syntax error, or zero rows all come back as a tool message the model can
read and act on:

```python
except UnsafeQueryError as exc:
    return f"REJECTED: {exc}"
```

That is what makes the loop useful rather than just expensive — the model
recovers from its own mistakes instead of the process dying. Zero rows in
particular returns a message telling it to go look up the entity's real name
before concluding the graph has no answer.

### Files

| File | Role |
| --- | --- |
| `config.py` | All settings from `.env`. No hardcoded credentials. |
| `graph.py` | `HetionetGraph`: schema introspection (cached to `.schema_cache.json`) + capped read-only execution. |
| `validate.py` | Read-only enforcement. |
| `examples.py` | 10 few-shot question → Cypher pairs. |
| `agent.py` | The prompt template, the `run_cypher` tool, the LCEL chain around `create_react_agent`, and `ask_traced()`. |
| `main.py` | CLI. Core logic is importable, so a FastAPI wrapper needs no rewrite. |
| `observability.py` | Langfuse tracing. Every function is a no-op when the keys are unset. |
| `judge.py` | The LLM judge: `Judgment` schema, per-category rubrics, structured-output chain. |
| `metrics.py` | Rule-based signals, gold `Expectation` checks, and the summary table. |
| `evaluate.py` | 21-question eval set across 5 categories, scored three ways. |
| `test_offline.py` | 60 assertions covering validation, `<think>` stripping, the execution guardrails, the tracing callback, the metrics and the judge schema. Never calls the LLM, so it needs no Groq key and spends no tokens. |

### Why not `langchain_neo4j.Neo4jGraph` / `GraphCypherQAChain`?

The public Hetionet endpoint runs **Neo4j 3.5.12** (Bolt 3.0, no APOC).
`Neo4jGraph` always sends a database name, and Bolt 3.0 rejects that outright:

```
ConfigurationError: Database name parameter for selecting database is not
supported in Bolt Protocol 3.0. Database name 'neo4j'.
```

So `graph.py` is a ~130-line wrapper over the official driver that works on both
3.5 and 5.x. Everything else is stock LangGraph. Neo4j 3.5 also means no `CALL {}`
subqueries and no `EXISTS {}`, which the system prompt tells the model explicitly —
otherwise it writes modern Cypher that the server rejects. Point `NEO4J_URI` at
your own Neo4j 5.x instance and it still works — set `NEO4J_DATABASE` too.

## Guardrails

| Guardrail | Where |
| --- | --- |
| Read-only clause blocklist (`CREATE`, `DELETE`, `MERGE`, `SET`, `REMOVE`, `DROP`, `DETACH`, `LOAD CSV`, `FOREACH`) | `validate.py` — string literals are blanked first, so `WHERE name = 'heart set'` is not a false positive |
| Procedure allowlist (only `db.labels` / `db.relationshipTypes` / `db.schema`) | `validate.py` |
| Bolt read access mode — server refuses writes even if the regex is fooled | `graph.py` |
| 10s query timeout | `graph.py`, `QUERY_TIMEOUT_SECONDS` |
| 25-row cap, enforced whether or not the LLM emitted a `LIMIT` | `graph.py`, `MAX_ROWS` |
| Bounded agent loop — at most 5 queries per question, then a graceful message | `agent.py`, `MAX_QUERIES` |
| No hardcoded credentials; `.env` is gitignored | `config.py`, `.gitignore` |

## Observability

Every run can emit a [Langfuse](https://langfuse.com) trace — the full tree of
LLM calls and `run_cypher` invocations, with latency and token usage per step:

```
hetionet_text2cypher            4.4s   1,842 tok
├── ChatGroq                    1.1s     612 tok
├── run_cypher                  0.3s          MATCH (d:Disease) RETURN count(d) …
└── ChatGroq                    0.9s     498 tok   "The graph contains 137 diseases."
```

**Tracing is entirely optional.** With `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
unset, `observability.py` returns empty callback lists and nothing is sent — the
agent behaves exactly as it did before. If the backend is unreachable it logs a
warning and the answer still comes back; a tracing outage is not an outage.

```bash
# Free keys at https://cloud.langfuse.com, or self-host:
# https://langfuse.com/self-hosting/docker-compose
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

In the REPL the `thread_id` doubles as the Langfuse **session id**, so a follow-up
conversation groups into one session and you can read the whole exchange in order.

## Evaluation

`evaluate.py` runs 21 questions across five categories — simple 1-hop, multi-hop,
aggregate, ambiguous entity naming, and questions with no answer in the graph.
It used to record the answers and leave you to read all 21 yourself. Now each one
is scored three ways:

| Evidence | Cost | What it catches |
| --- | --- | --- |
| **Deterministic** (`metrics.py`) | free | Did it run any Cypher? Were queries rejected? Did it burn the whole query budget, or return zero rows? |
| **Gold expectations** (`metrics.py`) | free | A wrong query that returns *real* rows — invisible to groundedness, because the summary of those rows is perfectly faithful. |
| **LLM judge** (`judge.py`) | ~1.5k tokens/question | Claims the rows do not support, and whether the category's rubric was met. |

```bash
python evaluate.py                          # all 21, judged
python evaluate.py --only ambiguous         # one category
python evaluate.py --no-judge               # rule-based metrics only, no judge tokens
python evaluate.py --judge-only results.json  # re-score saved runs, no agent calls
```

```
category       n  ground   relev  expect    gold  cypher   qrys   tokens
--------------------------------------------------------------------------
aggregate      2    5.00    5.00    1.00    1.00    1.00   1.00     9330
no-answer      2    3.00    5.00    0.50    0.50    1.00   1.00     8820
```

### The judge

The agent's system prompt makes a checkable promise — *"Use ONLY data returned by
the tool. Never add biomedical facts from your own knowledge"* — and nothing
verified it. The judge does: it sees the Cypher, the rows those queries actually
returned, and the answer, then scores **groundedness** and **relevance** 1-5 plus a
per-category `expectation_met`, and lists every unsupported claim by name.

A fact that is *true in the real world but absent from the rows* still scores 1.
That is the point: on "Who discovered penicillin?" the model knows the answer from
its own weights, and supplying it is the exact failure mode being tested.

```
[1/2] (no-answer) Who discovered penicillin?
  answer   : Penicillin was discovered by Alexander Fleming in 1928 at St Marys...
  judged   : grounded 1/5  relevant 5/5  expected NO
  UNSUPPORTED: Alexander Fleming, 1928 at St Marys Hospital in London
```

Two things keep the judge honest. It runs on a **different model** than the agent,
since a model grading its own output rates it generously (and Groq's token cap is
per model, so the judge gets its own budget). And `metrics.cross_check` flags any
answer the judge *passed* that had no rows behind it and did not decline — a
rubber-stamp, surfaced as `SUSPECT` rather than quietly averaged in.

Judgments are attached to the Langfuse trace as scores, so the table above and the
timeline in the UI are the same data.

### Gold expectations

Only a subset of questions carry one, because an expectation is only sound where
the answer is stable. Results are capped at `MAX_ROWS`, so a question whose full
result set exceeds the cap returns an arbitrary 25 of *N* rows — naming entities
there would fail at random rather than on a regression. `epilepsy syndrome` has
exactly 25 treating compounds, and the aggregates are ordered, so those qualify:

```python
"How many diseases are in the graph?": Expectation(number=137),
"Which 10 compounds treat the most diseases?": Expectation(entities=["Methotrexate"]),
```

Every `no-answer` question carries `Expectation(decline=True)` instead — a
zero-token regex check that the answer actually reports finding nothing.

One eval question is *mis-categorised* by intent, not by accident: "What drugs
treat heart attack?" sits under `no-answer` rather than `ambiguous`, because
Hetionet has no `myocardial infarction` node at all — only `coronary artery
disease`. Translating the term and then reporting no results is correct here.

### Known limits

- **Cost scales with the loop.** Every iteration re-sends the full schema and the
  conversation so far, so a question needing 3 queries costs roughly 4x one that
  needs 1. Questions with no answer are the worst case: the agent spends its
  whole `MAX_QUERIES` budget hunting for an alternate name before giving up.
  Lower `MAX_QUERIES` to cap this.
- **A hard "no" is expensive to reach.** "What drugs treat heart attack?" burns
  all 5 queries — `myocardial`, `infarction`, `heart`, `ischemia` — because
  Hetionet genuinely has no such node (the nearest is `coronary artery disease`).
  That is correct behaviour, just not cheap.
- **Groq free tier caps at 100k tokens/day, per model.** `LLM_MAX_RETRIES`
  handles brief 429s, not the daily cap — use `--only` to re-run a single
  category, or switch `GROQ_MODEL` to get a fresh budget:

  ```bash
  GROQ_MODEL=qwen/qwen3.8-27b python main.py "What treats hypertension?"
  ```

  Reasoning models such as `qwen/qwen3.8-27b` wrap their chain of thought in
  `<think>` tags; `_strip_reasoning()` removes it so only the answer is printed.

## The Hetionet schema

11 node labels — `Anatomy`, `BiologicalProcess`, `CellularComponent`, `Compound`,
`Disease`, `Gene`, `MolecularFunction`, `Pathway`, `PharmacologicClass`,
`SideEffect`, `Symptom` — and 24 relationship types, all abbreviated
(`TREATS_CtD`, `ASSOCIATES_DaG`, `BINDS_CbG`, `CAUSES_CcSE`, …). The agent fetches
this at startup, so the prompt always matches the live graph.

Node names are exact strings from source ontologies, so the prompt instructs
case-insensitive `CONTAINS` matching — a user typing "heart attack" still needs to
reach `myocardial infarction`, which is what the `ambiguous` eval category probes.
