"""
Microsoft Agent Framework — parallel supervisor multi-agent workflow.

Architecture:
  query → [fan-out] → financial | legal_risk | technical | summary (parallel)
                               → [fan-in] → Synthesizer → final answer

Note: Microsoft Agent Framework uses acyclic DAGs — no loop-back cycles.
The supervisor pattern is implemented as parallel fan-out to all specialists,
then fan-in to a synthesizer, which is functionally equivalent for most use cases.

Docs: https://learn.microsoft.com/en-us/agent-framework/workflows/agents-in-workflows
pip install agent-framework-openai

Public interface (used by routes.py):
  MODEL_ID : str
  stream(messages) → AsyncGenerator[str, None]
"""

import os

from typing import AsyncGenerator, List, Optional

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentResponseUpdate,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from agent_framework.openai import OpenAIChatCompletionClient

from tools import rag

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-ms-supervisor"
GEMINI_MODEL = "models/gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

SPECIALISTS = {
    "financial": "revenue income profit margin financial statements cash flow",
    "legal_risk": "risk factors legal proceedings regulatory compliance liabilities",
    "technical": "product technology research development innovation strategy",
    "summary": "overview business strategy key highlights performance",
}

# ── Shared state keys ─────────────────────────────────────────────────────────

QUERY_KEY = "query"
OUTPUTS_KEY = "specialist_outputs"
SPECIALIST_COUNT = len(SPECIALISTS)

# ── Shared client ─────────────────────────────────────────────────────────────

client = OpenAIChatCompletionClient(
    model=GEMINI_MODEL,
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url=GEMINI_BASE_URL,
)

# ── Tools ─────────────────────────────────────────────────────────────────────

from agent_framework import tool


@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)


@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )


# ── Entry executor — stores query in shared state, fans out ───────────────────


@executor(id="entry")
async def entry(message: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    ctx.set_state(QUERY_KEY, message)
    ctx.set_state(OUTPUTS_KEY, {})
    ctx.set_state("aggregator_fired", False)
    await ctx.send_message(
        AgentExecutorRequest(
            messages=[Message(role="user", contents=[message])], should_respond=True
        )
    )


# ── Specialist AgentExecutors ─────────────────────────────────────────────────


def make_specialist(name: str) -> AgentExecutor:
    focus = SPECIALISTS[name]
    return AgentExecutor(
        client.as_agent(
            name=name,
            instructions=(
                f"You are a {name} specialist. Focus on: {focus}. "
                "If the query mentions a company name, year, or document type, call get_filter_context first. "
                "Always call search_documents to retrieve relevant context. "
                "Answer using only retrieved context. Bold all figures using **value**. "
                "Cite source filenames. Never use code blocks."
            ),
            tools=[get_filter_context, search_documents],
        ),
        id=f"specialist_{name}",
    )


specialists = {name: make_specialist(name) for name in SPECIALISTS}

# ── Collector executors — save each specialist's output to shared state ───────


def make_collector(name: str) -> object:
    @executor(id=f"collect_{name}")
    async def collect(response: AgentExecutorResponse, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        outputs: dict = ctx.get_state(OUTPUTS_KEY) or {}
        outputs[name] = response.agent_response.text
        
        ctx.set_state(OUTPUTS_KEY, outputs)
        query: str = ctx.get_state(QUERY_KEY) or ""
        
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[Message(role="user", contents=[query])], should_respond=True
            )
        )

    return collect


collectors = {name: make_collector(name) for name in SPECIALISTS}

# ── Aggregator — fires synthesizer only when all specialists have reported ────


@executor(id="aggregator")
async def aggregator(_request: AgentExecutorRequest, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    outputs: dict = ctx.get_state(OUTPUTS_KEY) or {}
    if len(outputs) < SPECIALIST_COUNT:
        return  # not all specialists done yet
    if ctx.get_state("aggregator_fired"):
        return  # already fired — prevent duplicate synthesizer calls
    ctx.set_state("aggregator_fired", True)
    query: str = ctx.get_state(QUERY_KEY) or ""
    combined = "\n\n".join(f"## {name.upper()} ANALYSIS\n{text}" for name, text in outputs.items())
    
    await ctx.send_message(
        AgentExecutorRequest(
            messages=[
                Message(
                    role="user",
                    contents=[f"Query: {query}\n\nSpecialist Analyses:\n{combined}"],
                )
            ],
            should_respond=True,
        )
    )


# ── Synthesizer ───────────────────────────────────────────────────────────────

synthesizer_exec = AgentExecutor(
    client.as_agent(
        name="Synthesizer",
        instructions=(
            "Synthesize the specialist analyses into one comprehensive, well-structured answer. "
            "Cite source documents. Bold all specific numbers, percentages, dates, and key figures using **value**. "
            "Never use code blocks or backticks. "
            "References format: '1. filename, p.XX'"
        ),
    ),
    id="synthesizer",
)

# ── Workflow graph ─────────────────────────────────────────────────────────────


def build_workflow():
    builder = WorkflowBuilder(start_executor=entry)
    for name in SPECIALISTS:
        builder.add_edge(entry, specialists[name])
        builder.add_edge(specialists[name], collectors[name])
        builder.add_edge(collectors[name], aggregator)
    builder.add_edge(aggregator, synthesizer_exec)
    return builder.build()


workflow = build_workflow()

# ── Helpers ────────────────────────────────────────────────────────────────────


def last_user_message(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


# ── Public interface ───────────────────────────────────────────────────────────


async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    async for event in workflow.run(last_user_message(messages), stream=True):
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            if event.data.author_name == "Synthesizer" and event.data.text:
                yield event.data.text
