"""
sql_tool_factory.py
--------------------
I built one shared "factory" function instead of copy-pasting the same
SQL-agent code three times for HeartDiseaseDBTool, CancerDBTool, and
DiabetesDBTool. Each tool file just calls `make_db_tool(...)` with its
own database path, table name, tool name, and description.

How each tool answers a question (matches the assignment requirement:
"Execute user questions via SQL, return the result in natural
language"):
    1. The natural-language question comes in (e.g. "what is the
       average age of patients with heart disease?").
    2. LangChain's `create_sql_query_chain` turns it into a real SQL
       query using the LLM + the database schema.
    3. I execute that SQL query against the SQLite database (READ-ONLY
       connection — I never allow INSERT/UPDATE/DELETE).
    4. I feed the question + SQL query + SQL result back into the LLM
       so it replies in plain natural language instead of a raw table.
"""

from pathlib import Path

from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import get_llm

ANSWER_PROMPT = PromptTemplate.from_template(
    """Given the following user question, the corresponding SQL query, \
and the SQL result, answer the user's question in clear, natural \
language. Do not mention that you ran a SQL query.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)


class DBToolInput(BaseModel):
    question: str = Field(description="A natural language question about the dataset's statistics or records.")


def _forbid_write_queries(sql: str) -> str:
    """A simple safety guard so the LLM can never generate a write query."""
    forbidden = ("insert", "update", "delete", "drop", "alter", "create")
    if sql.strip().lower().startswith(forbidden):
        raise ValueError("Write operations are not permitted on this read-only tool.")
    return sql


def make_db_tool(db_path: Path, table_name: str, tool_name: str, description: str) -> StructuredTool:
    """
    Builds a ready-to-use LangChain tool bound to one SQLite database.

    Args:
        db_path: path to the .db file (e.g. data/db/heart_disease.db)
        table_name: the single table this tool is scoped to
        tool_name: name shown to the agent (e.g. "HeartDiseaseDBTool")
        description: tells the routing agent WHEN to pick this tool
    """
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", include_tables=[table_name])
    llm = get_llm()

    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDatabaseTool(db=db)

    chain = (
        RunnablePassthrough.assign(query=write_query | _forbid_write_queries)
        .assign(result=lambda x: execute_query.invoke(x["query"]))
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    def _run(question: str) -> str:
        return chain.invoke({"question": question})

    return StructuredTool.from_function(
        func=_run,
        name=tool_name,
        description=description,
        args_schema=DBToolInput,
    )
