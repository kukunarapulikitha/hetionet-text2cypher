"""Langfuse tracing, kept entirely optional.

Every function here is a no-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
are unset, so cloning the repo and running it with only a Groq key behaves
exactly as it did before tracing existed. ``langfuse`` is imported lazily
inside the functions to keep that true even if the package is missing.

Trace ids are minted up front by ``new_trace_id()`` and handed to the handler
via ``trace_context``, rather than read back off the handler afterwards. That
is what lets evaluate.py attach judge scores to the right trace: the id exists
before the run starts, so scoring never has to guess.
"""

import logging

import config

log = logging.getLogger("hetionet")

_SCORE_KIND = {bool: "BOOLEAN", str: "CATEGORICAL"}


def enabled() -> bool:
    """True when both Langfuse keys are configured.

    Checked before the handler is ever constructed — building one without
    credentials logs an authentication error that would confuse anyone who
    simply never opted into tracing.
    """
    return bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)


def new_trace_id(seed: str | None = None) -> str | None:
    """Mint a trace id to attach both spans and scores to."""
    if not enabled():
        return None
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=seed)


def _client():
    from langfuse import get_client

    return get_client()


def callbacks(trace_id: str | None = None) -> list:
    """LangChain callbacks that stream the run to Langfuse, or [] when off.

    A handler is bound to one trace, so callers building a specific trace pass
    its id and get a handler dedicated to it.
    """
    if not enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler

        context = {"trace_id": trace_id} if trace_id else None
        return [CallbackHandler(trace_context=context)]
    except Exception as exc:  # noqa: BLE001 - tracing must never break a run
        log.warning("Langfuse tracing disabled: %s", exc)
        return []


def metadata(session_id: str | None = None, tags: list[str] | None = None) -> dict:
    """Langfuse's reserved metadata keys, read off the runnable config.

    session_id groups a multi-turn conversation into one session in the UI,
    which is why main.py passes the REPL's thread_id straight through.
    """
    if not enabled():
        return {}
    meta: dict = {"langfuse_environment": config.LANGFUSE_TRACING_ENVIRONMENT}
    if session_id:
        meta["langfuse_session_id"] = session_id
    if tags:
        meta["langfuse_tags"] = tags
    return meta


def record_scores(
    trace_id: str | None,
    scores: dict[str, float | bool | str],
    comment: str | None = None,
) -> None:
    """Attach evaluation scores to an existing trace, one score per dimension."""
    if not enabled() or not trace_id or not scores:
        return
    try:
        client = _client()
        for name, value in scores.items():
            client.create_score(
                name=name,
                value=value,
                trace_id=trace_id,
                data_type=_SCORE_KIND.get(type(value), "NUMERIC"),
                comment=comment,
            )
    except Exception as exc:  # noqa: BLE001 - scoring must never fail the eval
        log.warning("Could not record Langfuse scores: %s", exc)


def flush() -> None:
    """Send anything still buffered. Both entry points are short-lived CLIs,
    so without this the last trace of a run is simply lost."""
    if not enabled():
        return
    try:
        _client().flush()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not flush Langfuse events: %s", exc)


def trace_url(trace_id: str | None) -> str | None:
    if not enabled() or not trace_id:
        return None
    return f"{config.LANGFUSE_BASE_URL.rstrip('/')}/trace/{trace_id}"
