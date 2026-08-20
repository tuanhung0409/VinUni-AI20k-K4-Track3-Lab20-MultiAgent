"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces a polished final synthesis with strict source citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        sources_list_text = (
            "\n".join(
                f"[[Source {idx + 1}]] {doc.title} | URL: {doc.url or 'N/A'}"
                for idx, doc in enumerate(state.sources)
            )
            or "None available"
        )

        system_prompt = (
            "You are an Elite Technical Research Writer. Your task is to craft a comprehensive, "
            "rigorous, and engaging research report synthesizing the gathered research and "
            "analysis.\n\n"
            "Formatting & Citation Rules:\n"
            "1. Grounding & Citations: Every major claim, metric, and finding MUST cite its "
            "corresponding source using bracketed citations, e.g. [[Source 1]], [[Source 2]].\n"
            "2. Structure:\n"
            "   - Executive Summary\n"
            "   - Core Architecture & Technical Foundations\n"
            "   - Comparative & Trade-off Analysis\n"
            "   - Source Reliability & Provenance Audit\n"
            "   - Practical Recommendations & Future Outlook\n"
            "   - References (A structured bibliography linking back to all sources)\n"
            "3. Tone: Professional, authoritative, and structured in GitHub Markdown with clear "
            "headings and bullet points."
        )
        user_prompt = (
            f"Topic: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Available Sources Catalog:\n{sources_list_text}\n\n"
            f"Researcher Notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Analyst Notes:\n{state.analysis_notes or 'N/A'}\n\n"
            "Write the complete, publication-ready research report following all citation rules."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.final_answer = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"citations_included": True},
            )
        )
        state.add_trace_event(
            name="writer_completed",
            payload={
                "final_answer_length": len(response.content),
                "tokens_input": response.input_tokens,
                "tokens_output": response.output_tokens,
            },
        )
        return state
