# Architecture

## Overview

AI Agentic Studio is built on a layered architecture:

1. **User Interface**: Streamlit frontend.
2. **Orchestration Layer**: LangGraph workflow engine.
3. **Agent Layer**: CrewAI agents with tools.
4. **Knowledge Layer**: ChromaDB for RAG.
5. **Infrastructure**: SQLite for reporting, optional Prometheus for monitoring.

## Data Flow

1. User submits a research topic and configuration.
2. The **Planner** agent generates a research plan.
3. Multiple **Researcher** agents execute the plan using web search, scraping, news API, and RAG.
4. The **Writer** compiles findings into a structured whitepaper.
5. The **Quality Gate** scores readability and depth; if insufficient, the Writer is retried.
6. The **Fact Checker** reviews for inaccuracies.
7. The **Citation Manager** adds references.
8. (Optional) The **Translator** converts the report to another language.
9. Final report is saved, exported, and optionally emailed.

## State Management

All state is maintained in a LangGraph `AgentState` TypedDict, passed through each node. No persistent memory across sessions.

## Observability

LangSmith callbacks are injected into the graph execution to trace every step. Additionally, structured logs are written to stdout.

## Error Recovery

- Retries with backoff on network/API errors.
- Fallback to default text if report generation fails.
- Timeout per agent team to prevent infinite loops.
