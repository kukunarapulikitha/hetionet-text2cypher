"""Thin read-only Neo4j wrapper: schema introspection + capped query execution.

We deliberately do not use ``langchain_neo4j.Neo4jGraph``. It always sends a
database name, which Bolt 3.0 (Neo4j 3.5, what the public Hetionet endpoint
runs) rejects outright. This wrapper works on both 3.5 and 5.x.
"""

import json
import os
from typing import Any

from neo4j import READ_ACCESS, GraphDatabase

import config


class SchemaError(RuntimeError):
    """Raised when the graph schema cannot be introspected."""


class HetionetGraph:
    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD,
        database: str | None = config.NEO4J_DATABASE,
    ) -> None:
        # The public Hetionet server has auth disabled; passing credentials to
        # it is an error, so send auth=None when no password is configured.
        auth = (user, password) if password else None
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._database = database
        self._schema: str | None = None
        self._labels: set[str] = set()
        self._relationship_types: set[str] = set()
        self._triples: set[tuple[str, str, str]] = set()

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "HetionetGraph":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- session plumbing -------------------------------------------------
    def _session(self):
        kwargs: dict[str, Any] = {"default_access_mode": READ_ACCESS}
        if self._database:
            kwargs["database"] = self._database
        return self._driver.session(**kwargs)

    def _read(
        self,
        cypher: str,
        timeout: float | None = None,
        max_rows: int | None = None,
        params: dict | None = None,
    ) -> list[dict]:
        """Run a read query, optionally truncating at ``max_rows``.

        Schema introspection passes ``max_rows=None`` because it must see every
        row; user-facing queries pass the configured cap.
        """
        with self._session() as session:
            tx = session.begin_transaction(timeout=timeout)
            try:
                result = tx.run(cypher, params or {})
                rows = []
                for record in result:
                    if max_rows is not None and len(rows) >= max_rows:
                        result.consume()
                        break
                    rows.append(_serialize(record.data()))
                return rows
            finally:
                tx.close()

    # -- public API -------------------------------------------------------
    def query(self, cypher: str) -> list[dict]:
        """Execute a validated read query with a timeout and row cap.

        The row cap applies regardless of whether the query carried a LIMIT, so
        a missing LIMIT cannot flood the context window.
        """
        return self._read(
            cypher,
            timeout=config.QUERY_TIMEOUT_SECONDS,
            max_rows=config.MAX_ROWS,
        )

    def explain(self, cypher: str) -> None:
        """Server-side syntax/semantic check. Raises on a bad query."""
        with self._session() as session:
            session.run(f"EXPLAIN {cypher}").consume()

    @property
    def schema(self) -> str:
        self._ensure_schema()
        return self._schema  # type: ignore[return-value]

    @property
    def labels(self) -> set[str]:
        """Every node label in the graph, for validating generated queries."""
        self._ensure_schema()
        return self._labels

    @property
    def relationship_types(self) -> set[str]:
        self._ensure_schema()
        return self._relationship_types

    @property
    def triples(self) -> set[tuple[str, str, str]]:
        """(start_label, rel_type, end_label) triples defining the metagraph."""
        self._ensure_schema()
        return self._triples

    def refresh_schema(self) -> str:
        """Re-introspect the graph and rewrite the on-disk cache."""
        schema, labels, rel_types, triples = self._build_schema()
        self._schema, self._labels = schema, labels
        self._relationship_types, self._triples = rel_types, triples
        with open(config.SCHEMA_CACHE_PATH, "w") as fh:
            json.dump(
                {
                    "schema": schema,
                    "labels": sorted(labels),
                    "relationship_types": sorted(rel_types),
                    "triples": sorted(triples),
                },
                fh,
                indent=2,
            )
        return schema

    def _ensure_schema(self) -> None:
        if self._schema is None and not self._load_cached_schema():
            self.refresh_schema()

    def _load_cached_schema(self) -> bool:
        if not os.path.exists(config.SCHEMA_CACHE_PATH):
            return False
        try:
            with open(config.SCHEMA_CACHE_PATH) as fh:
                cached = json.load(fh)
            # Older caches held only the schema string; treat those as stale so
            # the label/rel-type sets are always populated.
            required = ("schema", "labels", "relationship_types", "triples")
            if not all(k in cached for k in required):
                return False
            self._schema = cached["schema"]
            self._labels = set(cached["labels"])
            self._relationship_types = set(cached["relationship_types"])
            self._triples = {tuple(t) for t in cached["triples"]}
            return True
        except (OSError, json.JSONDecodeError, TypeError):
            return False

    # -- schema construction ----------------------------------------------
    def _build_schema(self) -> tuple[str, set[str], set[str], set[tuple[str, str, str]]]:
        try:
            labels = [r["label"] for r in self._read("CALL db.labels()")]
            rel_types = [
                r["relationshipType"] for r in self._read("CALL db.relationshipTypes()")
            ]
            triples = self._metagraph()
            properties = self._node_properties()
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise SchemaError(f"Could not introspect graph schema: {exc}") from exc

        lines = ["Node labels:"]
        lines += [f"  :{label}" for label in sorted(labels)]

        lines.append("")
        lines.append("Node properties:")
        for label in sorted(properties):
            props = ", ".join(sorted(properties[label]))
            lines.append(f"  :{label} -> {props}")

        lines.append("")
        lines.append("Relationship types (the Hetionet metagraph):")
        if triples:
            for start, rel, end in triples:
                lines.append(f"  (:{start})-[:{rel}]->(:{end})")
        else:
            # Fallback when db.schema.visualization() is unavailable.
            lines += [f"  :{rel}" for rel in sorted(rel_types)]

        return "\n".join(lines), set(labels), set(rel_types), set(triples)

    def _metagraph(self) -> list[tuple[str, str, str]]:
        """(start_label, rel_type, end_label) triples via db.schema.visualization()."""
        try:
            rows = self._read("CALL db.schema.visualization()")
        except Exception:  # noqa: BLE001 - optional enrichment
            return []
        if not rows:
            return []
        triples = set()
        for rel in rows[0].get("relationships") or []:
            # Driver returns each relationship as (start_node, type, end_node).
            try:
                start, rel_type, end = rel
                triples.add((start["name"], rel_type, end["name"]))
            except (TypeError, ValueError, KeyError):
                continue
        return sorted(triples)

    def _node_properties(self) -> dict[str, set[str]]:
        try:
            rows = self._read("CALL db.schema.nodeTypeProperties()")
        except Exception:  # noqa: BLE001 - optional enrichment
            return {}
        properties: dict[str, set[str]] = {}
        for row in rows:
            name = row.get("propertyName")
            if not name:
                continue
            for label in row.get("nodeLabels") or []:
                properties.setdefault(label, set()).add(name)
        return properties


def _serialize(value: Any) -> Any:
    """Convert driver types (nodes, rels, temporals) into JSON-safe values."""
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
