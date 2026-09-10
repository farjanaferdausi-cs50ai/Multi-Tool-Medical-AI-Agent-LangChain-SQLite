"""
CancerDBTool
-------------
I use this tool whenever the user's question is about statistics,
numbers, or specific records inside the Cancer dataset (Wisconsin
Breast Cancer style: cell-characteristic measurements per sample) —
e.g. average clump thickness, how many samples have high bare_nuclei
values, etc.
"""

from src.config import CANCER_DB_PATH
from src.tools.sql_tool_factory import make_db_tool

cancer_tool = make_db_tool(
    db_path=CANCER_DB_PATH,
    table_name="cancer_records",
    tool_name="CancerDBTool",
    description=(
        "Use this tool to answer STATISTICAL or DATA questions about the "
        "Cancer dataset stored in cancer.db (columns include Id, "
        "clump_thickness, uniformity_of_cell_size, "
        "uniformity_of_cell_shape, marginal_adhesion, "
        "single_epithelial_cell_size, bare_nuclei, bland_chromatin, "
        "normal_nucleoli, mitosis). Examples: 'what is the average single "
        "epithelial cell size in the cancer data', 'what is the average "
        "clump thickness', 'how many samples have bare_nuclei greater "
        "than 5'. Do NOT use this tool for general medical definitions or "
        "symptoms — use MedicalWebSearchTool for those."
    ),
)

