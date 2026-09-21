"""
Agent setup - LLM, agent creation, and public interface.

Public interface (used by routes.py):
  MODEL_ID                 → str
  stream(messages) → AsyncGenerator[str, None]
"""

from typing import AsyncGenerator, List

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_filter_context, search_documents

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-agent"

SYSTEM_PROMPT = """
You are a helpful document assistant.
For complex questions — especially multi-company comparisons or multi-year analyses — break them into individual queries (one per company, one per year) and call search_documents separately for each before forming a combined answer.
If the query mentions a company name, year, or document type, call get_filter_context first to get the available metadata fields and filter suggestions.
Always call search_documents to find information before answering.
Never include raw filter data, JSON, or tool output in your final response — only use them internally to guide retrieval.
Never wrap your response in code blocks or backticks.
If no documents are found, say so honestly — never make up an answer.
Always mention the source document in your answer.
Bold all specific numbers, percentages, dates, and key financial figures using **value**.
If you include a References section, format it as a numbered list with one reference per line: '1. filename, p.XX'
"""

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=[get_filter_context, search_documents],
    system_prompt=SYSTEM_PROMPT,
)

# ── Public interface ──────────────────────────────────────────────────────────

async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    """Stream the agent's response token by token, yielding plain text chunks."""
    async for event in agent.astream_events(
        {"messages": messages}, version="v2"
    ):
        if event["event"] != "on_chat_model_stream":
            continue
        chunk = event["data"]["chunk"]
        if getattr(chunk, "tool_call_chunks", None):
            continue  # skip tool-call chunks
        text = chunk.text
        if text:
            yield text
