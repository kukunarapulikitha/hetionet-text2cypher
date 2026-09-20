# Hetionet plain-English query agent

Ask a biomedical question in English. The agent reads the Neo4j schema, writes the
Cypher, runs it against [Hetionet](https://het.io/), and answers in plain English.
No Cypher knowledge required.

Built with **LangGraph** + **ChatGroq**. Read-only by construction.

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
python test_offline.py                          # 19 assertions, no Groq key needed
python evaluate.py                              # run the 21-question eval set
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
| `agent.py` | The prompt template, the `run_cypher` tool, and the LCEL chain around `create_react_agent`. |
| `main.py` | CLI. Core logic is importable, so a FastAPI wrapper needs no rewrite. |
| `evaluate.py` | 21-question eval set across 5 categories. |
| `test_offline.py` | 19 assertions covering read-only validation, `<think>` stripping and the execution guardrails. Never calls the LLM, so it needs no Groq key and spends no tokens. |

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

## Evaluation

`evaluate.py` runs 21 questions across five categories — simple 1-hop, multi-hop,
aggregate, ambiguous entity naming, and questions with no answer in the graph —
and logs each answer to `eval_results.json`.

```bash
python evaluate.py                    # all 21
python evaluate.py --only ambiguous   # one category
```

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
  GROQ_MODEL=qwen/qwen3.6-27b python main.py "What treats hypertension?"
  ```

  Reasoning models such as `qwen/qwen3.6-27b` wrap their chain of thought in
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
