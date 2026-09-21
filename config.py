"""Environment-backed configuration. No credentials are hardcoded."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


# --- Neo4j ---------------------------------------------------------------
# The public Hetionet instance (bolt://neo4j.het.io:7687) runs Neo4j 3.5 with
# auth disabled. Leave NEO4J_PASSWORD blank to connect anonymously.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j.het.io:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Only sent to the server when set. Neo4j 3.5 (Bolt 3.0) cannot select a
# database, so this must stay empty for the public Hetionet endpoint.
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None

# --- Groq ----------------------------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def groq_api_key() -> str:
    """Read lazily so importing this module never requires a key."""
    return _require("GROQ_API_KEY")


# --- LLM-as-a-judge ------------------------------------------------------
# Deliberately a different model than GROQ_MODEL: a model scoring its own
# output rates it too highly, and Groq's daily token cap is per model, so the
# judge gets its own budget. Reasoning models make poor judges here — they
# blow the output-tokens-per-minute limit on the free tier.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "3"))
# Tool rows are pasted into the judge prompt; 5 queries x 25 rows of JSON would
# otherwise dwarf the rubric.
JUDGE_MAX_RESULT_CHARS = int(os.getenv("JUDGE_MAX_RESULT_CHARS", "1500"))
# Minimum 1-5 score counted as a pass in the summary table.
JUDGE_PASS_SCORE = int(os.getenv("JUDGE_PASS_SCORE", "4"))

# --- Langfuse (observability) --------------------------------------------
# Entirely optional. With no keys set, tracing is off and nothing is sent.
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
LANGFUSE_TRACING_ENVIRONMENT = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")

# --- Guardrails ----------------------------------------------------------
QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "10"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "25"))
# How many Cypher queries the agent may run for one question before giving up.
MAX_QUERIES = int(os.getenv("MAX_QUERIES", "5"))
# Transport-level retries for Groq rate limits.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))

SCHEMA_CACHE_PATH = os.getenv("SCHEMA_CACHE_PATH", ".schema_cache.json")
