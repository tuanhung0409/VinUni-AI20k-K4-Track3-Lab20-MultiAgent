"""LangGraph workflow implementation."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph with LangGraph."""

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

    def _supervisor_node(self, state: ResearchState) -> ResearchState:
        return self.supervisor.run(state)

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        return self.researcher.run(state)

    def _analyst_node(self, state: ResearchState) -> ResearchState:
        return self.analyst.run(state)

    def _writer_node(self, state: ResearchState) -> ResearchState:
        return self.writer.run(state)

    def _route_condition(self, state: ResearchState) -> str:
        """Evaluate next node from supervisor's decision and max iteration limit."""
        settings = get_settings()

        if state.iteration >= settings.max_iterations:
            return END

        if not state.route_history:
            return END

        next_route = state.route_history[-1]
        if next_route in ("researcher", "analyst", "writer"):
            return next_route

        return END

    def build(self) -> Any:
        """Create and wire the LangGraph StateGraph."""
        builder = StateGraph(ResearchState)

        # Add agent nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("analyst", self._analyst_node)
        builder.add_node("writer", self._writer_node)

        # Flow starting at supervisor
        builder.add_edge(START, "supervisor")

        # Supervisor conditionally routes to workers or ends
        builder.add_conditional_edges(
            "supervisor",
            self._route_condition,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )

        # Worker nodes return back to supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")

        return builder

    def compile(self) -> Any:
        """Compile the state graph into an executable runner."""
        return self.build().compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final ResearchState."""
        app = self.compile()
        result = app.invoke(state)
        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
