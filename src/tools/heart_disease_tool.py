"""
HeartDiseaseDBTool
-------------------
I use this tool whenever the user's question is about statistics,
numbers, or specific records inside the Heart Disease dataset
(e.g. average cholesterol, how many patients have chest pain type 2,
count of patients above a certain age with high blood pressure, etc.).
"""

from src.config import HEART_DB_PATH
from src.tools.sql_tool_factory import make_db_tool

heart_disease_tool = make_db_tool(
    db_path=HEART_DB_PATH,
    table_name="heart_disease_records",
    tool_name="HeartDiseaseDBTool",
    description=(
        "Use this tool to answer STATISTICAL or DATA questions about the "
        "Heart Disease dataset stored in heart_disease.db (columns include "
        "age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, "
        "oldpeak, slope, ca, thal, target). Examples: 'what is the average "
        "cholesterol level of patients', 'how many patients have heart "
        "disease', 'what is the average age of male patients with high "
        "blood pressure'. Do NOT use this tool for general medical "
        "definitions or symptoms — use MedicalWebSearchTool for those."
    ),
)
