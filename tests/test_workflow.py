"""Unit tests for LangGraph MultiAgentWorkflow."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_build_and_compile() -> None:
    workflow = MultiAgentWorkflow()
    compiled = workflow.compile()
    assert compiled is not None


def test_workflow_stop_condition_on_final_answer() -> None:
    workflow = MultiAgentWorkflow()
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        final_answer="Pre-computed answer",
    )
    result = workflow.run(state)
    assert result.final_answer == "Pre-computed answer"
    assert result.route_history == ["done"]
