# Golden dataset

21 questions across 5 categories, in [questions.json](questions.json). `evaluate.py`
loads this file; it is the single source of truth for what the agent is tested on.

Each entry looks like this:

```json
{
  "category": "aggregate",
  "question": "How many diseases are in the graph?",
  "reference_cypher": "MATCH (d:Disease) RETURN count(d) AS n",
  "reference_answer": "There are 137 diseases in Hetionet.",
  "expected": { "number": 137 }
}
```

| Field | Role |
| --- | --- |
| `reference_cypher` | The query a human would write. Absent on the two questions the graph cannot answer at all. |
| `reference_answer` | What a correct answer says, derived by **running** that query — never written from memory. Present on all 21. |
| `total_results` | Full result-set size, which determines whether an `expected` block is safe (see below). |
| `expected` | Machine-checked assertions. Optional. |

The `reference_answer` is fed to the judge, which is what lets it catch a **wrong query
that returns real rows** — the answer is then perfectly faithful to those rows, so
groundedness scores 5/5 and only the reference reveals the error.

The `expected` block is checked in plain Python at **zero token cost**:

| Field | Meaning | Checked by |
| --- | --- | --- |
| `entities` | every name must appear in the answer | substring, case-insensitive |
| `number` | the answer must state this exact count | regex, tolerates `1,552` |
| `decline` | the answer must report finding nothing | regex over refusal phrasings |

## Why only 10 of 21 carry an `expected` block

Results are capped at `MAX_ROWS` (25). A question whose full result set exceeds the cap
returns an *arbitrary* 25 of N rows, because the agent rarely writes an `ORDER BY`.
Naming expected entities there would fail at random rather than on a regression — a
flaky test is worse than no test.

So a question is only annotated when its answer is stable: the result set fits inside
the cap, or the query is inherently ordered (`ORDER BY count DESC LIMIT 10`).

Verified against the live graph on 2026-09-21:

| Question | Full result size | Annotated? |
| --- | --- | --- |
| compounds treating epilepsy syndrome | **25** — exactly at the cap, so all are returned | yes |
| symptoms of asthma | 26 — one over, so one is dropped at random | no |
| anatomies expressing BRCA1 | 34 | no |
| genes associated with Crohn's disease | 120 | no |
| side effects of Metformin | 139 | no |

## How the values were obtained

Run directly against `bolt://neo4j.het.io:7687`, not taken from a model:

```cypher
MATCH (d:Disease) RETURN count(d) AS n;
-- 137

MATCH (c:Compound)-[:TREATS_CtD]->(d:Disease)
RETURN c.name, count(d) AS n ORDER BY n DESC LIMIT 4;
-- Methotrexate 19, Doxorubicin 17, Prednisone 15, Epirubicin 14

MATCH (c:Compound)-[:CAUSES_CcSE]->(s:SideEffect)
RETURN s.name, count(c) AS n ORDER BY n DESC LIMIT 6;
-- Nausea 932, Headache 872, Dermatitis 866, Rash 865, Vomiting 856

MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'myocardial'
   OR toLower(d.name) CONTAINS 'infarction' RETURN d.name;
-- (no rows) — confirms "heart attack" belongs under no-answer, not ambiguous
```

Hetionet is a static dataset, so these do not rot. If you point `NEO4J_URI` at a
different graph, re-verify them before trusting a failure.

## Adding a question

Append to `questions.json`. Annotate only if you have run the Cypher yourself and the
result is stable under the row cap — otherwise leave `expected` off and let the judge
score it reference-free.
