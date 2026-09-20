"""Read-only enforcement for generated Cypher."""

import re

# Clause-level writes, schema changes, and anything that can reach the
# filesystem or the network. Matched on word boundaries, case-insensitively.
FORBIDDEN = (
    "CREATE",
    "DELETE",
    "DETACH",
    "MERGE",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD CSV",
    "FOREACH",
    "USING PERIODIC COMMIT",
)

_FORBIDDEN_RE = re.compile(
    "|".join(rf"\b{re.escape(word)}\b" for word in FORBIDDEN),
    re.IGNORECASE,
)

# Only these procedure namespaces may be CALLed. Everything else (dbms.*,
# apoc.* writers, gds.*) is refused.
_ALLOWED_CALL_RE = re.compile(r"\bCALL\s+(db\.(labels|relationshipTypes|schema)\b)", re.IGNORECASE)
_ANY_CALL_RE = re.compile(r"\bCALL\b", re.IGNORECASE)


class UnsafeQueryError(ValueError):
    """Raised when a generated query is not provably read-only."""


def _strip_literals(cypher: str) -> str:
    """Blank out string literals so words inside them don't trip the scanner.

    Without this, a legitimate query like ``WHERE d.name = 'heart set'`` would
    be rejected for containing SET.
    """
    return re.sub(r"'[^']*'|\"[^\"]*\"", "''", cypher)


def validate(cypher: str) -> str:
    """Return the query unchanged, or raise UnsafeQueryError explaining why not."""
    if not cypher or not cypher.strip():
        raise UnsafeQueryError("The query was empty. Produce a valid read-only Cypher query.")

    stripped = _strip_literals(cypher)

    found = sorted({m.group(0).upper() for m in _FORBIDDEN_RE.finditer(stripped)})
    if found:
        raise UnsafeQueryError(
            f"Rejected: the query contains write operation(s) {', '.join(found)}. "
            "This agent is strictly read-only. Rewrite it using only MATCH, "
            "OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY and LIMIT."
        )

    calls = _ANY_CALL_RE.findall(stripped)
    if calls and len(calls) != len(_ALLOWED_CALL_RE.findall(stripped)):
        raise UnsafeQueryError(
            "Rejected: CALL to a procedure that is not allowed. Answer the "
            "question with a plain MATCH ... RETURN query instead."
        )

    return cypher
