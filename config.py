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


# --- Guardrails ----------------------------------------------------------
QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "10"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "25"))
# How many Cypher queries the agent may run for one question before giving up.
MAX_QUERIES = int(os.getenv("MAX_QUERIES", "5"))
# Transport-level retries for Groq rate limits.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))

SCHEMA_CACHE_PATH = os.getenv("SCHEMA_CACHE_PATH", ".schema_cache.json")
