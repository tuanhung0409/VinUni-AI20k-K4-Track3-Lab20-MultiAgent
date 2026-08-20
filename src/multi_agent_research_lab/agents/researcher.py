"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes with provenance."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        sources = self.search_client.search(
            query=state.request.query, max_results=state.request.max_sources
        )
        state.sources = sources

        sources_text = "\n\n".join(
            f"[Source {idx + 1}] Title: {doc.title}\nURL: {doc.url or 'N/A'}\n"
            f"Credibility/Type: {doc.metadata.get('credibility', 'general')}\n"
            f"Snippet: {doc.snippet}"
            for idx, doc in enumerate(sources)
        )

        system_prompt = (
            "You are a specialized AI Research Agent. Your task is to investigate the research "
            "query using the retrieved sources.\n"
            "Guidelines:\n"
            "1. Extract essential factual claims, mechanisms, and key discoveries.\n"
            "2. Associate facts with specific source indices (e.g. [Source 1], [Source 2]).\n"
            "3. Identify core definitions, technological foundations, and empirical findings.\n"
            "4. Maintain strict fidelity to the sources; do not hallucinate unsupported claims."
        )
        user_prompt = (
            f"Research Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Retrieved Sources ({len(sources)} documents):\n"
            f"{sources_text}\n\n"
            "Please generate structured Research Notes organized by key themes, including "
            "citations to [Source X] for every claim."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={"sources_retrieved": len(sources)},
            )
        )
        state.add_trace_event(
            name="researcher_completed",
            payload={
                "sources_retrieved": len(sources),
                "notes_length": len(response.content),
                "tokens_input": response.input_tokens,
                "tokens_output": response.output_tokens,
            },
        )
        return state
