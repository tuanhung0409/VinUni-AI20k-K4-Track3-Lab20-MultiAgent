"""Observability and tracing hooks for LangSmith, Langfuse, and OpenTelemetry."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings


def setup_tracing() -> dict[str, str]:
    """Configure tracing environment variables for LangSmith / Langfuse if keys are set."""
    settings = get_settings()
    configured_providers: dict[str, str] = {}

    # 1. LangSmith auto-instrumentation for LangGraph & OpenAI
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        configured_providers["LangSmith"] = f"Project: {settings.langsmith_project}"

    # 2. Langfuse telemetry
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
        configured_providers["Langfuse"] = f"Host: {settings.langfuse_host}"

    return configured_providers


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for hierarchical tracing spans.

    Records span duration, attributes, status, and error metadata.
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "status": "started",
        "duration_seconds": None,
        "error": None,
    }
    try:
        yield span
        span["status"] = "completed"
    except Exception as exc:
        span["status"] = "failed"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
