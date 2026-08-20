"""Optional critic agent for fact-checking and report verification."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Fact-checking, citation audit, and hallucination review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append quality audit findings."""
        if not state.final_answer:
            return state

        sources_text = "\n".join(
            f"- [Source {idx + 1}]: {doc.title}" for idx, doc in enumerate(state.sources)
        )

        system_prompt = (
            "You are a rigorous Research Critic & Fact-Checker. Audit the final report against "
            "the provided sources, verify citations, and flag any ungrounded assertions."
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Final Report:\n{state.final_answer}\n\n"
            "Produce an audit scorecard checking:\n"
            "1. Citation Integrity\n"
            "2. Factual Consistency\n"
            "3. Grounding & Hallucination Assessment"
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
            )
        )
        state.add_trace_event(
            name="critic_completed",
            payload={"audit_length": len(response.content)},
        )
        return state
