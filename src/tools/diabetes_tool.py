"""
DiabetesDBTool
---------------
I use this tool whenever the user's question is about statistics,
numbers, or specific records inside the Diabetes dataset (Pima Indians
schema) — e.g. average glucose level, how many patients were diagnosed,
average age of diagnosed patients, etc.
"""

from src.config import DIABETES_DB_PATH
from src.tools.sql_tool_factory import make_db_tool

diabetes_tool = make_db_tool(
    db_path=DIABETES_DB_PATH,
    table_name="diabetes_records",
    tool_name="DiabetesDBTool",
    description=(
        "Use this tool to answer STATISTICAL or DATA questions about the "
        "Diabetes dataset stored in diabetes.db (columns include "
        "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
        "DiabetesPedigreeFunction, Age, Outcome). Examples: 'what is the "
        "average glucose level of diagnosed patients', 'how many patients "
        "were diagnosed with diabetes', 'what is the average age for those "
        "diagnosed with diabetes'. Do NOT use this tool for general "
        "medical definitions or symptoms — use MedicalWebSearchTool for "
        "those."
    ),
)
