"""
prepare_db.py
--------------
I use this script to convert the three raw medical CSV datasets
(Heart Disease, Cancer Prediction, Diabetes) into three separate
SQLite databases with meaningful table names and correctly mapped
column types (INTEGER / REAL / TEXT).

Usage:
    python src/prepare_db.py

Input (expected):
    data/csv/heart.csv
    data/csv/cancer.csv
    data/csv/diabetes.csv

Output:
    data/db/heart_disease.db   -> table: heart_disease_records
    data/db/cancer.db          -> table: cancer_records
    data/db/diabetes.db        -> table: diabetes_records

Note:
    The three placeholder CSVs shipped in data/csv/ use the exact
    same column names as the real Kaggle datasets referenced in the
    assignment. Once I download the real datasets, I only need to
    overwrite the CSV files in data/csv/ (same file names, same
    columns) and re-run this script — no code changes required.
"""

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "data" / "csv"
DB_DIR = BASE_DIR / "data" / "db"

# Mapping of pandas dtypes to SQLite column types
DTYPE_MAP = {
    "int64": "INTEGER",
    "float64": "REAL",
    "object": "TEXT",
    "bool": "INTEGER",
}

# Each dataset: (csv filename, sqlite db filename, table name)
DATASETS = [
    ("heart.csv", "heart_disease.db", "heart_disease_records"),
    ("cancer.csv", "cancer.db", "cancer_records"),
    ("diabetes.csv", "diabetes.db", "diabetes_records"),
]


def infer_sqlite_schema(df: pd.DataFrame) -> str:
    """I build a CREATE TABLE column definition string from a DataFrame's dtypes."""
    columns = []
    for col in df.columns:
        sqlite_type = DTYPE_MAP.get(str(df[col].dtype), "TEXT")
        safe_col = col.replace(" ", "_").replace("-", "_")
        columns.append(f'"{safe_col}" {sqlite_type}')
    return ", ".join(columns)


def build_database(csv_name: str, db_name: str, table_name: str) -> None:
    csv_path = CSV_DIR / csv_name
    db_path = DB_DIR / db_name

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found. Place the CSV there and re-run.")
        return

    df = pd.read_csv(csv_path)
    # Clean column names (spaces/dashes -> underscores) so SQL queries stay simple
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]

    schema_sql = infer_sqlite_schema(df)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    cursor.execute(f'CREATE TABLE "{table_name}" ({schema_sql})')
    conn.commit()

    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()

    row_count = cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    print(f"[OK] {db_name} -> table '{table_name}' created with {row_count} rows.")

    conn.close()


def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    for csv_name, db_name, table_name in DATASETS:
        build_database(csv_name, db_name, table_name)


if __name__ == "__main__":
    main()
