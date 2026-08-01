import pytest
from unittest.mock import patch, MagicMock
from app import planning_node, research_node, writing_node, quality_node, fact_check_node, citation_node, translation_node

def test_planning_node():
    state = {
        "topic": "AI",
        "num_agents": 2,
        "writing_style": "formal",
        "llm_provider": "Ollama",
        "llm_model": "phi3:mini"
    }
    with patch("app.LLMFactory.get_llm") as mock_llm:
        mock_llm.return_value = MagicMock()
        result = planning_node(state)
        assert "plan" in result
        assert result["status"] == "Structure Planned"

def test_quality_node_empty_report():
    state = {"final_report": "", "topic": "AI"}
    result = quality_node(state)
    assert result["quality_score"] >= 0
    assert "readability_scores" in result
    # Check that fallback was applied
    assert "Executive Summary" in state.get("final_report", "")
