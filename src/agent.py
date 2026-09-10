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

I build this with LangChain's tool-calling AgentExecutor, which is the
same underlying mechanism as the OpenAI Agent SDK: it relies on the
model's native function/tool-calling ability (OpenAI function calling)
to decide which tool to invoke and with what arguments. This matches
both my class note pattern and the assignment's "OpenAI Agent SDK +
LangChain Agent Executor" requirement.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
   - Cancer Prediction dataset -> CancerDBTool
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


def build_agent_executor() -> AgentExecutor:
    llm = get_llm()
    tools = [heart_disease_tool, cancer_tool, diabetes_tool, medical_web_search_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    # handle_parsing_errors=True -> resilience pattern from my class note,
    # prevents the whole app from crashing on a malformed tool call.
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def ask(question: str) -> str:
    """Convenience function: build the agent and ask it one question."""
    executor = build_agent_executor()
    response = executor.invoke({"input": question})
    return response["output"]
