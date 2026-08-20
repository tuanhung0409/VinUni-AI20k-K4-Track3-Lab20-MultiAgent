"""Supervisor / router agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect state and determine the next agent route.

        Routing policy:
        1. If max iterations reached or final answer exists -> 'done'
        2. If sources or research notes are missing -> 'researcher'
        3. If analysis notes are missing -> 'analyst'
        4. If final answer is missing -> 'writer'
        5. Otherwise -> 'done'
        """
        settings = get_settings()

        if state.iteration >= settings.max_iterations or state.final_answer is not None:
            next_route = "done"
        elif not state.sources and not state.research_notes:
            next_route = "researcher"
        elif not state.analysis_notes:
            next_route = "analyst"
        elif state.final_answer is None:
            next_route = "writer"
        else:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event(
            name="supervisor_decision",
            payload={
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources or state.research_notes),
                "has_analysis": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state
