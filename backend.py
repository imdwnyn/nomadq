import os
import operator
import json
import asyncio

from typing import Any, TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


load_dotenv()


# =========================
# LLM
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Please add it to your .env file."
    )


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# =========================
# State
# =========================

class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start:end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor + Guardrail
# =========================

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning
or travel information.

Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing,
or itineraries.

Block clearly unrelated requests and requests asking for harmful
or illegal instructions.

Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:

{{
    "allowed": true,
    "reason": ""
}}

User request:
{query}
"""

    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning "
            "application. Return strict JSON only.",
            guardrail_prompt,
        )

        guardrail_result = _json_from_llm(guardrail_raw)

        allowed = bool(
            guardrail_result.get("allowed", True)
        )

        guardrail_reason = str(
            guardrail_result.get("reason", "")
        ).strip()

        llm_calls += 1

    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")

        allowed = True

        guardrail_reason = (
            "Guardrail validation fallback allowed the request."
        )

    if not allowed:

        reason = guardrail_reason or (
            "NomadQ can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, "
            "weather, budget, or itinerary."
        )

        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(
                    content=f"Guardrail blocked request: {reason}"
                )
            ],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.

Choose only the specialist agents needed for the request.

Available agents:

- flight_agent: flights, airports, airlines, routes,
  airfare, or booking advice

- hotel_agent: hotels, accommodation, neighborhoods,
  or places to stay

- weather_agent: weather, climate, season, forecast,
  or packing advice

- budget_agent: cost, affordability, price limits,
  or budget feasibility

- itinerary_agent: creates the integrated travel plan
  and must always be included

Return strict JSON only using this schema:

{{
    "selected_agents": [
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent"
    ],
    "trip_constraints": {{
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": []
    }},
    "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. "
            "Return strict JSON only.",
            supervisor_prompt,
        )

        parsed = _json_from_llm(supervisor_raw)

        requested_agents = parsed.get(
            "selected_agents",
            []
        )

        selected_agents = [
            name
            for name in AGENT_ORDER
            if name in requested_agents
            and name in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()

        parsed_constraints = parsed.get(
            "trip_constraints",
            {}
        )

        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(
            parsed.get("reasoning", "")
        ).strip()

        llm_calls += 1

    except Exception as exc:

        print(f"Supervisor fallback used: {exc}")

        selected_agents = AGENT_ORDER.copy()

        constraints = _empty_constraints()

        reasoning = (
            "Supervisor parsing failed, so the original "
            "full travel workflow was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [
            AIMessage(
                content="Supervisor created the agent plan."
            )
        ],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================

def guardrail_blocked_agent(state: TravelState):

    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "This request was blocked by the travel input guardrail."
    )

    return {
        "final_response": reason,
        "messages": [
            AIMessage(content=reason)
        ],
    }

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

Analyze the user's travel request and provide useful flight
planning guidance.

User Request:
{query}

Airport Data:
{airport_data}

Airline Data:
{airline_data}

Provide:

1. Recommended route
2. Suitable airports
3. Relevant airlines
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

# =========================
# Temporary specialist nodes
# =========================

def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel flight planner."
                ),
                HumanMessage(content=prompt),
            ]
        )

        flight_data = response.content

    except Exception as exc:

        flight_data = (
            f"Flight information unavailable: {exc}"
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def hotel_agent(state: TravelState):

    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:

        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:

        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


def weather_agent(state: TravelState):

    city = extract_destination(
        state["user_query"]
    )

    try:

        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:

        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


def budget_agent(state: TravelState):

    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:

1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable,
clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a practical travel budget analyst."
            ),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [
            AIMessage(
                content="Budget assessment generated."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


def itinerary_agent(state: TravelState):

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical,
budget-aware, and easy to follow.

Create a clear draft that is ready
for human review.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an expert travel planner."
            ),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. "
        "Approve it to create the final polished plan, "
        "or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [
            AIMessage(
                content="Draft itinerary created for human review."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Dynamic Supervisor Routing
# =========================

ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:

    selected = state.get(
        "selected_agents",
        []
    )

    return [
        agent
        for agent in AGENT_ORDER
        if agent in selected
    ]


def route_from_supervisor(state: TravelState) -> str:

    if not state.get(
        "guardrail_allowed",
        True
    ):
        return "guardrail_blocked"

    selected = _selected_agents(state)

    return (
        selected[0]
        if selected
        else "itinerary_agent"
    )


def route_after_agent(current_agent: str):

    def route(state: TravelState) -> str:

        selected = _selected_agents(state)

        current_index = AGENT_ORDER.index(
            current_agent
        )

        for next_agent in AGENT_ORDER[
            current_index + 1:
        ]:

            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node(
    "supervisor",
    supervisor_agent
)

graph.add_node(
    "guardrail_blocked",
    guardrail_blocked_agent
)

graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "weather_agent",
    weather_agent
)

graph.add_node(
    "budget_agent",
    budget_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)


graph.add_edge(
    START,
    "supervisor"
)

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    ROUTE_MAP
)

graph.add_conditional_edges(
    "flight_agent",
    route_after_agent("flight_agent"),
    ROUTE_MAP
)

graph.add_conditional_edges(
    "hotel_agent",
    route_after_agent("hotel_agent"),
    ROUTE_MAP
)

graph.add_conditional_edges(
    "weather_agent",
    route_after_agent("weather_agent"),
    ROUTE_MAP
)

graph.add_conditional_edges(
    "budget_agent",
    route_after_agent("budget_agent"),
    ROUTE_MAP
)

graph.add_edge(
    "itinerary_agent",
    END
)

graph.add_edge(
    "guardrail_blocked",
    END
)


travel_graph = graph.compile()


# =========================
# Test
# =========================

def run_nomadq(query: str):

    initial_state = {
        "messages": [
            HumanMessage(content=query)
        ],
        "user_query": query,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "trip_constraints": _empty_constraints(),
        "supervisor_reasoning": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "llm_calls": 0,
    }

    return travel_graph.invoke(
        initial_state
    )


if __name__ == "__main__":

    result = run_nomadq(
        "Plan a 5 day trip to Mumbai from Dubai."
    )

    print("\n====================")
    print("       NomadQ")
    print("====================")

    print(
        "\nSelected agents:",
        result.get("selected_agents")
    )

    print(
        "\nTrip constraints:",
        result.get("trip_constraints")
    )

    print(
        "\nSupervisor reasoning:",
        result.get("supervisor_reasoning")
    )

    print(
        "\nItinerary:",
        result.get("itinerary")
    )

    print(
        "\nLLM calls:",
        result.get("llm_calls")
    )