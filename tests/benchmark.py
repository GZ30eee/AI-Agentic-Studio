import pytest
from app import app_graph, AgentState
import textstat

TOPICS = [
    "Quantum Computing in 2025",
    "Sustainable Supply Chains",
    "The Future of Remote Work",
    "AI in Healthcare Diagnostics",
    "Blockchain for Digital Identity"
]

def run_pipeline(topic):
    state = AgentState(
        topic=topic,
        email="",
        num_agents=2,
        writing_style="formal",
        llm_provider="Ollama",
        llm_model="phi3:mini",
        target_language=None,
        plan="",
        raw_research="",
        final_report="",
        quality_score=0.0,
        status="",
        citations=[],
        research_notes=[],
        fact_check_report="",
        readability_scores={},
        cache_key="",
        refinement_attempts=0,
        total_tokens=0,
        total_cost=0.0
    )
    # Run graph (simplified, just writer for speed)
    from app import writing_node, quality_node
    state = writing_node(state)
    state = quality_node(state)
    return state

def test_benchmark():
    results = []
    for topic in TOPICS:
        state = run_pipeline(topic)
        report = state.get("final_report", "")
        length = len(report)
        flesch = textstat.flesch_reading_ease(report)
        citations = state.get("citations", [])
        passed = (length > 1500 and flesch > 30 and len(citations) >= 3)
        results.append(passed)
        print(f"{topic}: length={length}, flesch={flesch}, citations={len(citations)} -> {'PASS' if passed else 'FAIL'}")
    success_rate = sum(results) / len(results) * 100
    print(f"\nBenchmark Success Rate: {success_rate:.2f}%")
    assert success_rate >= 60.0  # threshold
