"""
Supervisor multi-agent system with LangGraph.
A supervisor routes each query to relevant specialist agents, then synthesizes their outputs.

Specialists: financial, legal_risk, technical, summary
Supervisor decides which to call (up to 4 iterations), then synthesize → final answer.

Public interface (used by routes.py):
  MODEL_ID : str
  stream(messages) → AsyncGenerator[str, None]
"""

from typing import AsyncGenerator, Dict, List, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from tools import rag

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-supervisor"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

SPECIALISTS = {
    "financial":  "revenue income profit margin financial statements cash flow",
    "legal_risk": "risk factors legal proceedings regulatory compliance liabilities",
    "technical":  "product technology research development innovation strategy",
    "summary":    "overview business strategy key highlights performance",
}

SUPERVISOR_PROMPT = """
You manage specialized document analysis agents.

Agents: financial | legal_risk | technical | summary

Query: {query}
Already called: {called}
Outputs so far: {outputs}

Which agent to call next, or FINISH if you have enough information?
Rules: do not repeat an agent; FINISH when sufficient.
Respond with one word only: financial | legal_risk | technical | summary | FINISH
"""

# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str
    next_agent: str
    agent_outputs: Dict[str, str]
    final_answer: str
    iteration: int

# ── Nodes ─────────────────────────────────────────────────────────────────────

async def supervisor(state: State) -> State:
    called = list(state["agent_outputs"].keys())
    outputs = "\n".join(f"- {k}: {v[:100]}..." for k, v in state["agent_outputs"].items()) or "none"
    result = await llm.ainvoke([HumanMessage(SUPERVISOR_PROMPT.format(query=state["query"], called=called or "none", outputs=outputs))])
    decision = result.text.strip().lower()
    next_agent = decision if decision in SPECIALISTS else "FINISH"
    return {**state, "next_agent": next_agent, "iteration": state["iteration"] + 1}


def make_specialist(name: str):
    focus = SPECIALISTS[name]

    async def node(state: State) -> State:
        query = f"{focus} {state['query']}"
        filters = rag.extract_filters(query)
        context = rag.retrieve(query, filters=filters)
        if not context or context == "No relevant documents found.":
            output = f"No relevant {name} information found."
        else:
            result = await llm.ainvoke([HumanMessage(f"You are a {name} specialist.\nAnswer using only the provided context. Bold all figures using **value**.\nNever wrap your response in code blocks or backticks.\n\nQuery: {state['query']}\n\nContext:\n{context}")])
            output = result.text
        return {**state, "agent_outputs": {**state["agent_outputs"], name: output}}

    return node


async def synthesize(state: State) -> State:
    if not state["agent_outputs"]:
        return {**state, "final_answer": "No relevant information found."}
    combined = "\n\n".join(f"{k}: {v}" for k, v in state["agent_outputs"].items())
    result = await llm.ainvoke([HumanMessage(f"Synthesize these analyses into one comprehensive answer.\nBold all figures using **value**. Cite sources. Never use code blocks or backticks.\nReferences format: '1. filename, p.XX'\n\nQuery: {state['query']}\n\n{combined}")])
    return {**state, "final_answer": result.text}

# ── Routing ───────────────────────────────────────────────────────────────────

def route(state: State) -> Literal["financial", "legal_risk", "technical", "summary", "synthesize"]:
    if state["next_agent"] == "FINISH" or state["iteration"] >= 4:
        return "synthesize"
    return state["next_agent"]  # type: ignore[return-value]

# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("synthesize", synthesize)
    for name in SPECIALISTS:
        graph.add_node(name, make_specialist(name))
        graph.add_edge(name, "supervisor")
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor", route,
        {**{n: n for n in SPECIALISTS}, "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


graph = build_graph()

# ── Public interface ──────────────────────────────────────────────────────────

async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    async for event in graph.astream_events(State(query=query, next_agent="", agent_outputs={}, final_answer="", iteration=0), version="v2"):
        if event["event"] == "on_chat_model_stream" and event.get("metadata", {}).get("langgraph_node") == "synthesize":
            chunk = event["data"]["chunk"]
            if chunk.text:
                yield chunk.text
