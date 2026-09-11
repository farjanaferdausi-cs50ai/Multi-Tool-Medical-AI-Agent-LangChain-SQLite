"""
agent.py
--------
This is my Main AI Agent. It wires all four tools together and routes
every incoming question to the correct one:

    - HeartDiseaseDBTool   -> statistics/data questions about heart.csv
    - CancerDBTool         -> statistics/data questions about cancer.csv
    - DiabetesDBTool       -> statistics/data questions about diabetes.csv
    - MedicalWebSearchTool -> general medical knowledge (definitions,
                               symptoms, cures)

I build this with LangChain's modern `create_agent` (built on LangGraph).
I moved to this from the older `create_tool_calling_agent` +
`AgentExecutor` combo because Gemini's newer "thinking" models (3.x)
attach an internal "thought_signature" to every tool call, and it must
be echoed back to the API on the next turn. The old AgentExecutor
reconstructs messages when formatting its scratchpad and drops that
signature, causing a "Function call is missing a thought_signature"
error. `create_agent` keeps the original message objects intact across
turns, so the signature survives automatically.
"""

from langchain.agents import create_agent

from src.config import get_llm
from src.tools.cancer_tool import cancer_tool
from src.tools.diabetes_tool import diabetes_tool
from src.tools.heart_disease_tool import heart_disease_tool
from src.tools.web_search_tool import medical_web_search_tool

SYSTEM_PROMPT = """You are a professional medical data assistant with access to four tools.

Routing rules (follow strictly):
1. If the question asks about STATISTICS, DATA, NUMBERS, COUNTS, AVERAGES,
   or specific RECORDS from a dataset -> use the matching DB tool:
   - Heart Disease dataset -> HeartDiseaseDBTool
   - Cancer dataset -> CancerDBTool
   - Diabetes dataset -> DiabetesDBTool
2. If the question asks for a DEFINITION, SYMPTOM, CAUSE, RISK FACTOR, or
   CURE/TREATMENT (general medical knowledge, not tied to my dataset
   records) -> use MedicalWebSearchTool.
3. If a question mixes both (e.g. "what is diabetes and what is the
   average glucose level in the dataset"), call both tools and combine
   the answers clearly.
4. Never invent numbers. If a DB tool returns no result, say so honestly.
5. Always answer in clear, professional natural language — never show
   raw SQL or raw JSON to the user.
"""

TOOLS = [heart_disease_tool, cancer_tool, diabetes_tool, medical_web_search_tool]


def build_agent_executor():
    """Builds the compiled LangGraph agent. Call `.invoke(...)` on the result."""
    llm = get_llm()
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def ask(question: str) -> str:
    """Convenience function: build the agent and ask it one question."""
    agent = build_agent_executor()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
