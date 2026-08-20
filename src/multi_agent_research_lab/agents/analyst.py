"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into critical analysis, comparisons, and source evaluations."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        sources_meta = "\n".join(
            f"- [Source {idx + 1}]: {doc.title} (Provider: {doc.metadata.get('provider', 'N/A')}, "
            f"Credibility: {doc.metadata.get('credibility', 'standard')})"
            for idx, doc in enumerate(state.sources)
        )

        system_prompt = (
            "You are a Senior AI Systems Analyst. Your goal is to critically evaluate raw research "
            "notes and sources to extract deep architectural trade-offs, compare viewpoints, and "
            "assess source credibility.\n\n"
            "Requirements:\n"
            "1. Source Credibility Assessment: Evaluate the reliability and potential biases of "
            "the sources (e.g. peer-reviewed vs benchmark vs synthetic vs engineering reports).\n"
            "2. Comparative Analysis: Compare different architectures, paradigms, or approaches.\n"
            "3. Trade-off Matrix: Analyze performance vs cost vs latency vs complexity.\n"
            "4. Gap Analysis: Highlight weak evidence, missing claims, or unresolved questions."
        )
        user_prompt = (
            f"Research Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Sources Metadata:\n{sources_meta or 'No sources metadata.'}\n\n"
            f"Research Notes:\n{state.research_notes or 'No notes provided.'}\n\n"
            "Provide structured Analysis Notes with clear sections: "
            "1) Source Reliability & Evidence Quality, 2) Core Comparative Insights, "
            "3) Systems Trade-off Analysis, and 4) Limitations & Open Questions."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.analysis_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"has_source_assessment": True},
            )
        )
        state.add_trace_event(
            name="analyst_completed",
            payload={
                "analysis_length": len(response.content),
                "tokens_input": response.input_tokens,
                "tokens_output": response.output_tokens,
            },
        )
        return state
