# NomadQ 🌍

NomadQ is an autonomous multi-agent travel intelligence system built using **LangGraph**, the **Model Context Protocol (MCP)**, **Dynamic Supervisor Architecture**, and **Human-in-the-Loop (HITL)** validation gates.

## Core Architecture
- **Input Guardrail**: Deterministic evaluation of request safety and domain relevance.
- **Dynamic Supervisor Agent**: Intelligently routes tasks to specialist agents on demand.
- **MCP Tool Integration**: Decoupled tool server execution via FastMCP stdio.
- **Human-in-the-Loop (HITL)**: Workflow pauses via LangGraph `interrupt()` checkpoints before final itinerary generation.