"""
MedicalWebSearchTool
----------------------
I use this tool ONLY for general medical knowledge questions that my
three datasets cannot answer — definitions, symptoms, causes, cures,
prevention tips, etc. It is powered by the Tavily Search API (the same
provider I already used in my Module 22 Multi-Tool AI Agent project).

I deliberately restrict its description to "general medical knowledge
only" so the main routing agent never sends statistics/data questions
here — those belong to the three DB tools instead.
"""

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import TAVILY_API_KEY


class WebSearchInput(BaseModel):
    query: str = Field(
        description="A general medical knowledge question, e.g. a definition, symptom, cause, or cure."
    )


def _build_search_engine() -> TavilySearchResults:
    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY not found in .env. Get a free key at "
            "https://tavily.com and add it to your .env file."
        )
    return TavilySearchResults(api_key=TAVILY_API_KEY, max_results=4)


def _run_web_search(query: str) -> str:
    engine = _build_search_engine()
    results = engine.invoke({"query": f"{query} (medical information)"})
    if not results:
        return "I could not find reliable information for this query."

    formatted = []
    for r in results:
        content = r.get("content", "").strip()
        url = r.get("url", "")
        if content:
            formatted.append(f"- {content} (Source: {url})")

    return "\n".join(formatted) if formatted else "No relevant medical information found."


medical_web_search_tool = StructuredTool.from_function(
    func=_run_web_search,
    name="MedicalWebSearchTool",
    description=(
        "Use this tool ONLY for GENERAL medical knowledge questions such "
        "as definitions, symptoms, causes, risk factors, treatments, or "
        "cures (e.g. 'what are the symptoms of diabetes', 'what causes "
        "heart disease', 'how is breast cancer treated'). Do NOT use this "
        "tool for statistics or numeric questions about my datasets — use "
        "HeartDiseaseDBTool, CancerDBTool, or DiabetesDBTool for those."
    ),
    args_schema=WebSearchInput,
)
