import pytest
from app import app_graph, AgentState

def test_graph_routing():
    state = AgentState(
        topic="AI",
        email="",
        plan="plan",
        raw_research="research",
        final_report="",
        quality_score=1.5,
        status="",
        num_agents=2,
        agent_roles=[],
        writing_style="formal",
        llm_provider="Ollama",
        llm_model="phi3:mini",
        citations=[],
        research_notes=[],
        fact_check_report="",
        readability_scores={},
        target_language=None,
        cache_key="abc",
        refinement_attempts=1,
        total_tokens=0,
        total_cost=0.0
    )
    # Simulate quality gate routing
    from app import route_quality
    next_node = route_quality(state)
    assert next_node == "writer"  # score <1.8 and attempts<3
    state["quality_score"] = 2.0
    next_node = route_quality(state)
    assert next_node == "fact_check"
