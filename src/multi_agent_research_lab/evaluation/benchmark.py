"""Benchmark execution for single-agent vs multi-agent research architectures."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient

Runner = Callable[[str], ResearchState]

# Pricing for google/gemini-2.5-flash: $0.15 / 1M input tokens, $0.60 / 1M output tokens
PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60


def compute_metrics_from_state(
    state: ResearchState,
    latency_seconds: float,
    run_name: str,
) -> BenchmarkMetrics:
    """Calculate latency, cost, quality, citation coverage, and failure rate."""
    # 1. Failure rate
    is_failed = bool(state.errors) or not state.final_answer or len(state.final_answer.strip()) < 50
    failure_rate = 1.0 if is_failed else 0.0

    # 2. Token cost calculation from trace
    total_input_tokens = 0
    total_output_tokens = 0
    for event in state.trace:
        payload = event.get("payload", {})
        total_input_tokens += payload.get("tokens_input", 0) or payload.get("input_tokens", 0) or 0
        total_output_tokens += (
            payload.get("tokens_output", 0) or payload.get("output_tokens", 0) or 0
        )

    estimated_cost_usd = (total_input_tokens / 1_000_000.0) * PRICE_PER_M_INPUT + (
        total_output_tokens / 1_000_000.0
    ) * PRICE_PER_M_OUTPUT

    # 3. Citation coverage
    final_text = state.final_answer or ""
    num_sources = len(state.sources)
    if num_sources == 0:
        citation_coverage = 1.0 if not is_failed else 0.0
    else:
        cited_count = 0
        for idx, source in enumerate(state.sources):
            source_tag = f"Source {idx + 1}"
            bracket_tag = f"[{source_tag}]"
            short_title = source.title[:30].lower()
            if (
                source_tag.lower() in final_text.lower()
                or bracket_tag.lower() in final_text.lower()
                or (source.url and source.url in final_text)
                or (short_title and short_title in final_text.lower())
            ):
                cited_count += 1
        citation_coverage = min(1.0, cited_count / num_sources)

    # 4. Quality score (0 to 10 scale)
    if is_failed:
        quality_score = 0.0
    else:
        score = 5.0  # Base score for valid output
        word_count = len(final_text.split())
        if word_count >= 500:
            score += 2.0
        elif word_count >= 200:
            score += 1.0

        # Check structured sections
        has_sections = sum(
            1
            for heading in ["summary", "architecture", "trade-off", "reference", "conclusion"]
            if re.search(rf"#+\s*.*{heading}", final_text, re.IGNORECASE)
        )
        score += min(2.0, has_sections * 0.5)

        # Bonus for good citations
        score += citation_coverage * 1.0
        quality_score = round(min(10.0, score), 1)

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency_seconds, 2),
        estimated_cost_usd=round(estimated_cost_usd, 6),
        quality_score=quality_score,
        citation_coverage=round(citation_coverage, 2),
        failure_rate=failure_rate,
        notes=f"Tokens: {total_input_tokens} in / {total_output_tokens} out",
    )


def run_single_agent_baseline(query: str) -> ResearchState:
    """Execute single-agent baseline."""
    state = ResearchState(request=ResearchQuery(query=query))
    llm_client = LLMClient()
    system_prompt = (
        "You are an expert AI research assistant. Provide a comprehensive, accurate, "
        "and well-structured research report for the query."
    )
    response = llm_client.complete(system_prompt=system_prompt, user_prompt=query)
    state.final_answer = response.content
    state.add_trace_event(
        name="baseline_completion",
        payload={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    )
    return state


def run_multi_agent_workflow(query: str) -> ResearchState:
    """Execute multi-agent LangGraph workflow."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return execution state and calculated metrics."""
    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))

    latency = perf_counter() - started
    metrics = compute_metrics_from_state(state, latency_seconds=latency, run_name=run_name)
    return state, metrics
