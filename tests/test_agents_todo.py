"""Unit tests for SupervisorAgent routing policy."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_empty() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()
    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"


def test_supervisor_routes_to_analyst_when_sources_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[
            SourceDocument(
                title="Doc 1",
                url="https://example.com/1",
                snippet="Some snippet",
                provider="mock",
            )
        ],
    )
    supervisor = SupervisorAgent()
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_when_analysis_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Found relevant info.",
        analysis_notes="Analyzed comparison points.",
    )
    supervisor = SupervisorAgent()
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"


def test_supervisor_routes_to_done_when_final_answer_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        final_answer="Final synthesis completed.",
    )
    supervisor = SupervisorAgent()
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"
