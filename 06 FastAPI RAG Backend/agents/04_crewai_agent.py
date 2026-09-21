"""
CrewAI agent implementation.

Public interface (same contract as all agent files):
  MODEL_ID : str
  stream(messages) → AsyncGenerator[str, None]
"""

from typing import AsyncGenerator, List, Optional

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool as crewai_tool

from tools import rag

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-crewai"

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = LLM(model="gemini/gemini-2.5-flash")

# ── Tools ─────────────────────────────────────────────────────────────────────


@crewai_tool("get_filter_context")
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)


@crewai_tool("search_documents")
# def search_documents(query: str) -> str:
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = Agent(
    role="Document Assistant",
    goal="Answer user questions accurately using the document knowledge base.",
    backstory="""
                You are an expert document analyst.
                For complex questions — especially multi-company comparisons or multi-year analyses — break them into individual queries (one per company, one per year) and search for each separately before forming a combined answer.
                If the query mentions a company name, year, or document type, call get_filter_context first.
                You always search the knowledge base before answering and cite sources.
                Bold all specific numbers, percentages, dates, and key financial figures using **value**.
                Never wrap your response in code blocks or backticks.
                If no documents are found, you say so honestly.
                """,
    tools=[get_filter_context, search_documents],
    llm=llm,
    verbose=False,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def last_user_message(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


# ── Public interface ──────────────────────────────────────────────────────────
async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    task = Task(
        description=last_user_message(messages),
        expected_output="A detailed answer with source citations.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False, stream=True)
    streaming = await crew.akickoff()
    async for chunk in streaming:
        if chunk.content and "TOOL" not in str(chunk.chunk_type).upper():
            yield chunk.content
