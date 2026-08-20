"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    run_benchmark,
    run_multi_agent_workflow,
    run_single_agent_baseline,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline with an end-to-end LLM call."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    system_prompt = (
        "You are an expert AI research assistant. Your task is to provide a comprehensive, "
        "well-structured, factual, and in-depth research report answering the user's query."
    )

    try:
        llm_client = LLMClient()
        with console.status("[bold green]Executing single-agent baseline LLM call...[/bold green]"):
            response = llm_client.complete(system_prompt=system_prompt, user_prompt=request.query)

        state.final_answer = response.content
        state.add_trace_event(
            name="baseline_llm_completion",
            payload={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        console.print(
            Panel(state.final_answer, title="Single-Agent Baseline Result", border_style="green")
        )
    except Exception as exc:
        console.print(
            Panel(
                f"Error during baseline execution: {exc}",
                title="Execution Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))

    settings = get_settings()
    if settings.langsmith_api_key:
        console.print(
            Panel(
                f"[bold cyan]LangSmith Tracing Active![/bold cyan]\n"
                f"Project: [bold]{settings.langsmith_project}[/bold]\n"
                f"Dashboard: [link=https://smith.langchain.com/]https://smith.langchain.com/[/link]",
                title="Observability: LangSmith Trace UI",
                border_style="cyan",
            )
        )
    if settings.langfuse_public_key:
        console.print(
            Panel(
                f"[bold magenta]Langfuse Tracing Active![/bold magenta]\n"
                f"Host: [bold]{settings.langfuse_host}[/bold]\n"
                f"Dashboard: [link=https://cloud.langfuse.com/]https://cloud.langfuse.com/[/link]",
                title="Observability: Langfuse Trace UI",
                border_style="magenta",
            )
        )


@app.command("benchmark")
def benchmark(
    config_path: Annotated[
        str, typer.Option("--config", "-c", help="Path to config yaml")
    ] = "configs/lab_default.yaml",
    output_path: Annotated[
        str, typer.Option("--output", "-o", help="Output path for benchmark report")
    ] = "reports/benchmark_report.md",
) -> None:
    """Run comparative benchmark across queries for baseline vs multi-agent."""
    _init()
    queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    cfg_file = Path(config_path)
    if cfg_file.exists():
        try:
            with open(cfg_file, encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f)
            custom_queries = cfg_data.get("benchmark", {}).get("queries", [])
            if custom_queries:
                queries = custom_queries
        except Exception as err:
            console.print(f"[yellow]Warning reading config {config_path}: {err}[/yellow]")

    console.print(
        Panel.fit(
            f"Running benchmark on {len(queries)} queries comparing Single-Agent vs Multi-Agent",
            title="Benchmark Suite",
            style="bold cyan",
        )
    )

    metrics_list: list[BenchmarkMetrics] = []

    for idx, q in enumerate(queries, start=1):
        console.print(f"\n[bold yellow]Query {idx}/{len(queries)}: {q}[/bold yellow]")

        # 1. Run Baseline
        with console.status(f"[cyan]Running Single-Agent Baseline (Query {idx})...[/cyan]"):
            _, base_metric = run_benchmark(
                run_name=f"Q{idx}_Baseline",
                query=q,
                runner=run_single_agent_baseline,
            )
        metrics_list.append(base_metric)
        console.print(
            f"  [green][OK] Baseline finished in {base_metric.latency_seconds:.2f}s "
            f"(Quality: {base_metric.quality_score}/10, "
            f"Cost: ${base_metric.estimated_cost_usd:.5f})[/green]"
        )

        # 2. Run Multi-Agent
        with console.status(f"[magenta]Running Multi-Agent Workflow (Query {idx})...[/magenta]"):
            _, multi_metric = run_benchmark(
                run_name=f"Q{idx}_MultiAgent",
                query=q,
                runner=run_multi_agent_workflow,
            )
        metrics_list.append(multi_metric)
        console.print(
            f"  [green][OK] Multi-Agent finished in {multi_metric.latency_seconds:.2f}s "
            f"(Quality: {multi_metric.quality_score}/10, "
            f"Citation: {multi_metric.citation_coverage:.0%}, "
            f"Cost: ${multi_metric.estimated_cost_usd:.5f})[/green]"
        )

    # Render report
    report_content = render_markdown_report(metrics_list)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_content, encoding="utf-8")

    # Display Rich summary table
    table = Table(title="Benchmark Results Summary")
    table.add_column("Run Name", style="cyan")
    table.add_column("Latency", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Citation Cov.", justify="right")
    table.add_column("Failure Rate", justify="right")

    for m in metrics_list:
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}s",
            f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd else "$0.00",
            f"{m.quality_score}/10" if m.quality_score else "N/A",
            f"{m.citation_coverage:.0%}" if m.citation_coverage is not None else "N/A",
            f"{m.failure_rate:.0%}" if m.failure_rate is not None else "0%",
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]Report successfully generated at: {output_path}[/bold green]")


if __name__ == "__main__":
    app()
