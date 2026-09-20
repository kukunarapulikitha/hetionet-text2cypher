"""Few-shot question -> Cypher pairs, rendered by LangChain's FewShotPromptTemplate.

These teach four things the schema alone does not: Hetionet's abbreviated
relationship names, that entity names must be matched case-insensitively (users
type "heart attack", the graph says "myocardial infarction"), that Neo4j 3.5 has
no CALL {} subqueries, and that every query ends with a LIMIT.

The examples render into the *system* message rather than as human/ai turns.
That is deliberate: FewShotChatMessagePromptTemplate would emit AI messages whose
content is bare Cypher, teaching a tool-calling model to reply with query text
instead of calling run_cypher. The string form keeps them as reference material.
"""

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What compounds treat epilepsy syndrome?",
        "cypher": """MATCH (c:Compound)-[:TREATS_CtD]->(d:Disease)
WHERE toLower(d.name) CONTAINS toLower('epilepsy syndrome')
RETURN c.name AS compound, d.name AS disease
LIMIT 25""",
    },
    {
        "question": "Which genes are associated with Crohn's disease?",
        "cypher": """MATCH (d:Disease)-[:ASSOCIATES_DaG]->(g:Gene)
WHERE toLower(d.name) CONTAINS toLower("Crohn's disease")
RETURN g.name AS gene
LIMIT 25""",
    },
    {
        "question": "What side effects does Flavoxate cause?",
        "cypher": """MATCH (c:Compound)-[:CAUSES_CcSE]->(s:SideEffect)
WHERE toLower(c.name) CONTAINS toLower('Flavoxate')
RETURN s.name AS side_effect
LIMIT 25""",
    },
    {
        "question": "How many diseases are in the graph?",
        "cypher": """MATCH (d:Disease)
RETURN count(d) AS disease_count""",
    },
    {
        "question": "How many labels, nodes and relationships are there?",
        # Neo4j 3.5 has no CALL {} subqueries, so the counts chain through WITH.
        "cypher": """CALL db.labels() YIELD label
WITH count(label) AS label_count
MATCH (n)
WITH label_count, count(n) AS node_count
MATCH ()-[r]->()
RETURN label_count, node_count, count(r) AS relationship_count""",
    },
    {
        "question": "Which compounds bind genes that are associated with multiple sclerosis?",
        "cypher": """MATCH (d:Disease)-[:ASSOCIATES_DaG]->(g:Gene)<-[:BINDS_CbG]-(c:Compound)
WHERE toLower(d.name) CONTAINS toLower('multiple sclerosis')
RETURN DISTINCT c.name AS compound, g.name AS gene
LIMIT 25""",
    },
    {
        "question": "What symptoms does asthma present?",
        "cypher": """MATCH (d:Disease)-[:PRESENTS_DpS]->(s:Symptom)
WHERE toLower(d.name) CONTAINS toLower('asthma')
RETURN s.name AS symptom
LIMIT 25""",
    },
    {
        "question": "Which genes do the drugs that treat hypertension share?",
        "cypher": """MATCH (c:Compound)-[:TREATS_CtD]->(d:Disease)
WHERE toLower(d.name) CONTAINS toLower('hypertension')
MATCH (c)-[:BINDS_CbG]->(g:Gene)
WITH g, count(DISTINCT c) AS drug_count
WHERE drug_count > 1
RETURN g.name AS gene, drug_count
ORDER BY drug_count DESC
LIMIT 25""",
    },
    {
        "question": "What pathways do BRCA1-interacting genes participate in?",
        "cypher": """MATCH (g:Gene)-[:INTERACTS_GiG]-(other:Gene)-[:PARTICIPATES_GpPW]->(p:Pathway)
WHERE g.name = 'BRCA1'
RETURN DISTINCT p.name AS pathway
LIMIT 25""",
    },
    {
        "question": "Which 10 compounds treat the most diseases?",
        "cypher": """MATCH (c:Compound)-[:TREATS_CtD]->(d:Disease)
RETURN c.name AS compound, count(DISTINCT d) AS disease_count
ORDER BY disease_count DESC
LIMIT 10""",
    },
    {
        "question": "What anatomical structures is lung cancer localized to?",
        "cypher": """MATCH (d:Disease)-[:LOCALIZES_DlA]->(a:Anatomy)
WHERE toLower(d.name) CONTAINS toLower('lung cancer')
RETURN a.name AS anatomy
LIMIT 25""",
    },
]

# Braces inside the Cypher values are safe: PromptTemplate substitutes them in,
# it does not re-parse the substituted text for variables.
FEW_SHOT = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=PromptTemplate.from_template("Question: {question}\nCypher:\n{cypher}"),
    prefix="",
    suffix="",
    input_variables=[],
    example_separator="\n\n",
)


def format_examples() -> str:
    return FEW_SHOT.format().strip()
