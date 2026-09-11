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
    1. The natural-language question comes in.
    2. I ask the LLM to write a SQLite SELECT query for it, given the
       table schema (I write this prompt myself instead of relying on
       LangChain's old `create_sql_query_chain` helper, which was
       removed in LangChain 1.x).
    3. I execute that SQL query against the SQLite database (READ-ONLY
       — I never allow INSERT/UPDATE/DELETE/DROP/ALTER).
    4. I feed the question + SQL query + SQL result back into the LLM
       so it replies in plain natural language instead of a raw table.
"""

from pathlib import Path

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import get_llm

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    """You are a SQLite expert. Given an input question and a table \
schema, write a syntactically correct, READ-ONLY SQLite SELECT query \
that answers the question.

Table schema:
{table_info}

Question: {question}

Return ONLY the raw SQL query. No explanation, no markdown formatting, \
no code fences, no trailing semicolon commentary.
SQL Query:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Given the following user question, the corresponding SQL query, \
and the SQL result, answer the user's question in clear, natural \
language. Do not mention that you ran a SQL query.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""
)


class DBToolInput(BaseModel):
    question: str = Field(description="A natural language question about the dataset's statistics or records.")


def _clean_sql(raw: str) -> str:
    """Strips markdown fences (if any) and blocks write operations."""
    sql = raw.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:]
        sql = sql.strip()

    forbidden = ("insert", "update", "delete", "drop", "alter", "create")
    if sql.lower().startswith(forbidden):
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
    execute_query = QuerySQLDatabaseTool(db=db)
    table_info = db.get_table_info()

    def _run(question: str) -> str:
        llm = get_llm()
        sql_chain = SQL_GENERATION_PROMPT | llm | StrOutputParser()
        raw_sql = sql_chain.invoke({"question": question, "table_info": table_info})
        sql = _clean_sql(raw_sql)
        result = execute_query.invoke(sql)

        answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        return answer_chain.invoke({"question": question, "query": sql, "result": result})

    return StructuredTool.from_function(
        func=_run,
        name=tool_name,
        description=description,
        args_schema=DBToolInput,
    )
