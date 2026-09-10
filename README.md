# Multi-Tool AI Agent for Medical Datasets and Web Search

A multi-tool AI agent that answers statistical questions from three medical
SQLite databases (Heart Disease, Cancer Prediction, Diabetes) and routes
general medical knowledge questions (definitions, symptoms, cures) to a
live web search tool. Built with LangChain's tool-calling `AgentExecutor`
on top of OpenAI-compatible function calling.

## Overview

I built this project as my Module 23 assignment. It demonstrates how a
single AI agent can intelligently decide, per question, whether to:

1. Query a structured SQL database and answer with real statistics, or
2. Search the web for general medical knowledge.

## Architecture

```
                        ┌────────────────────────┐
                        │      Main AI Agent      │
                        │ (LangChain AgentExecutor)│
                        └────────────┬─────────────┘
                                     │ routes based on question type
        ┌────────────────┬──────────┼──────────┬────────────────┐
        ▼                ▼          ▼          ▼                ▼
HeartDiseaseDBTool  CancerDBTool  DiabetesDBTool        MedicalWebSearchTool
        │                │          │                            │
        ▼                ▼          ▼                            ▼
heart_disease.db     cancer.db  diabetes.db              Tavily Search API
(SQLite)             (SQLite)   (SQLite)                 (general knowledge)
```

**Routing logic:**
- Statistics / data / numbers questions → matching DB tool
- Definition / symptom / cure questions → `MedicalWebSearchTool`

## Project Structure

```
medical-multi-tool-agent/
├── data/
│   ├── csv/                     # raw CSV datasets (replace with real Kaggle files)
│   │   ├── heart.csv
│   │   ├── cancer.csv
│   │   └── diabetes.csv
│   └── db/                      # generated SQLite databases
│       ├── heart_disease.db
│       ├── cancer.db
│       └── diabetes.db
├── src/
│   ├── config.py                 # env loading + LLM initialization
│   ├── prepare_db.py              # CSV -> SQLite conversion script
│   ├── agent.py                   # main routing agent
│   └── tools/
│       ├── sql_tool_factory.py    # shared SQL-agent builder
│       ├── heart_disease_tool.py  # HeartDiseaseDBTool
│       ├── cancer_tool.py         # CancerDBTool
│       ├── diabetes_tool.py       # DiabetesDBTool
│       └── web_search_tool.py     # MedicalWebSearchTool (Tavily)
├── main.py                        # CLI chat entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Datasets Used

| Dataset | SQLite DB | Table | Key Columns |
|---|---|---|---|
| Heart Disease | `heart_disease.db` | `heart_disease_records` | age, sex, cp, trestbps, chol, thalach, target ... |
| Cancer (cell measurements) | `cancer.db` | `cancer_records` | clump_thickness, uniformity_of_cell_size, bare_nuclei, mitosis ... |
| Diabetes (Pima Indians) | `diabetes.db` | `diabetes_records` | Pregnancies, Glucose, BMI, Age, Outcome ... |

> The CSV files in `data/csv/` are the real datasets I downloaded from
> Kaggle for this assignment (Heart: 303 rows, Cancer: 101 rows,
> Diabetes: 768 rows). If I ever need to swap in an updated version of
> a dataset, I just replace the matching CSV file in `data/csv/` and
> re-run `python src/prepare_db.py` — no code changes required as long
> as the column names stay the same.

## How to Run

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/medical-multi-tool-agent.git
cd medical-multi-tool-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in **one** LLM option:
- Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`), **or**
- `OPENAI_API_KEY` for direct OpenAI access, **or**
- `GITHUB_TOKEN` (free, from [GitHub Settings → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens))

Also fill in:
- `TAVILY_API_KEY` (free tier at [tavily.com](https://tavily.com)) — required for `MedicalWebSearchTool`

### 3. (Optional) Update with newer Kaggle data

The real datasets are already included in `data/csv/`. If I want to
refresh them later, I download the updated CSVs and overwrite the
files with the same names (`heart.csv`, `cancer.csv`, `diabetes.csv`).

### 4. Build the SQLite databases

```bash
python src/prepare_db.py
```

This creates `heart_disease.db`, `cancer.db`, and `diabetes.db` inside
`data/db/` with correctly typed columns (INTEGER / REAL / TEXT).

### 5. Run the agent

```bash
python main.py
```

## Sample Questions

**Statistics (routed to a DB tool):**
- "What is the average cholesterol level in the heart disease dataset?"
- "What is the average single epithelial cell size in the cancer data?"
- "What is the average age of patients diagnosed with diabetes?"

**General medical knowledge (routed to MedicalWebSearchTool):**
- "What are the common symptoms of heart disease?"
- "What causes type 2 diabetes?"
- "How is breast cancer typically treated?"

**Mixed (routed to both):**
- "What is diabetes, and what is the average glucose level in the dataset?"

## Safety Notes

- All three DB tools connect in **read-only** query mode and explicitly
  block write operations (`INSERT` / `UPDATE` / `DELETE` / `DROP` /
  `ALTER`) so the agent can never modify the underlying data.
- API keys are loaded from a local `.env` file (never committed —
  see `.gitignore`).

## Key Frameworks Used

- **LangChain** — tool-calling `AgentExecutor`, SQL query chains
- **OpenAI-compatible function calling** — via GitHub Models or the OpenAI API
- **SQLite** — lightweight structured storage for all three datasets
- **Tavily Search API** — general medical knowledge retrieval

## Author

Built by Farjana Ferdausi as part of the Ostad AI/ML Engineering & Data
Science program (Module 23).
