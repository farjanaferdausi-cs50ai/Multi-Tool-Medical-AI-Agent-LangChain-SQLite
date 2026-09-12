# 🧠 Multi-Tool AI Agent for Medical Datasets and Web Search

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/LLM-Gemini-4285F4?logo=googlegemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/SQLite-Read--Only-07405E?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Search-Tavily-FF6B4A" alt="Tavily">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

A multi-tool AI agent that answers statistical questions from three medical
SQLite databases (Heart Disease, Cancer, Diabetes) and routes general
medical knowledge questions (definitions, symptoms, cures) to a live web
search tool. Built with LangChain's modern `create_agent` (LangGraph-based)
running on Google Gemini.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Datasets Used](#-datasets-used)
- [How to Run](#-how-to-run)
- [Sample Questions](#-sample-questions)
- [Safety Notes](#-safety-notes)
- [Key Frameworks Used](#-key-frameworks-used)
- [Author](#-author)

---

## 🔎 Overview

I built this project as my **Module 23 assignment** (Ostad AI/ML Engineering
& Data Science Program). It demonstrates how a single AI agent can
intelligently decide, per question, whether to:

1. 📊 Query a structured SQL database and answer with real statistics, or
2. 🌐 Search the web for general medical knowledge.

## 🏗️ Architecture

```
                        ┌────────────────────────┐
                        │      Main AI Agent      │
                        │  (LangChain create_agent) │
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
| Question type | Routed to |
|---|---|
| 📊 Statistics / data / numbers | Matching DB tool |
| 📖 Definition / symptom / cure | `MedicalWebSearchTool` |
| 🔀 Mixed question | Both tools, combined |

## 📁 Project Structure

```
medical-multi-tool-agent/
├── data/
│   ├── csv/                       # real Kaggle datasets
│   │   ├── heart.csv
│   │   ├── cancer.csv
│   │   └── diabetes.csv
│   └── db/                        # generated SQLite databases
│       ├── heart_disease.db
│       ├── cancer.db
│       └── diabetes.db
├── src/
│   ├── config.py                  # env loading + LLM initialization
│   ├── prepare_db.py               # CSV -> SQLite conversion script
│   ├── agent.py                    # main routing agent
│   └── tools/
│       ├── sql_tool_factory.py     # shared SQL-agent builder
│       ├── heart_disease_tool.py   # HeartDiseaseDBTool
│       ├── cancer_tool.py          # CancerDBTool
│       ├── diabetes_tool.py        # DiabetesDBTool
│       └── web_search_tool.py      # MedicalWebSearchTool (Tavily)
├── main.py                         # CLI chat entry point
├── requirements.txt
├── .env.example
└── README.md
```

## 📊 Datasets Used

| Dataset | SQLite DB | Table | Key Columns | Rows |
|---|---|---|---|---|
| 🫀  Heart Disease | `heart_disease.db` | `heart_disease_records` | age, sex, cp, trestbps, chol, thalach, target ... | 303 |
| 🧬 Cancer (cell measurements) | `cancer.db` | `cancer_records` | clump_thickness, uniformity_of_cell_size, bare_nuclei, mitosis ... | 101 |
| 🩸 Diabetes (Pima Indians) | `diabetes.db` | `diabetes_records` | Pregnancies, Glucose, BMI, Age, Outcome ... | 768 |

> The CSV files in `data/csv/` are the real datasets I downloaded from
> Kaggle for this assignment. If I ever need to swap in an updated version
> of a dataset, I just replace the matching CSV file in `data/csv/` and
> re-run `python src/prepare_db.py` — no code changes required as long as
> the column names stay the same.

## 🚀 How to Run

### 1. Clone and set up the environment

```bash
git clone https://github.com/farjanaferdausi-cs50ai/Multi-Tool-Medical-AI-Agent-LangChain-SQLite.git
cd Multi-Tool-Medical-AI-Agent-LangChain-SQLite
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🔑 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in **one** LLM option:
- `GOOGLE_API_KEY` **(recommended)** — free key from [Google AI Studio](https://aistudio.google.com/apikey). Google's key format changed in 2026 — new keys start with `AQ.` (the older `AIzaSy...` format is being phased out), both work with this project. Also set `GOOGLE_MODEL_NAME=gemini-3.5-flash-lite` (this is the default if left unset), **or**
- Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`), **or**
- `OPENAI_API_KEY` for direct OpenAI access

> ⚠️ **Note:** GitHub Models (`GITHUB_TOKEN`) was **permanently retired by
> GitHub on July 30, 2026** and no longer works. I switched this project's
> primary LLM backend to Google Gemini for that reason.

Also fill in:
- `TAVILY_API_KEY` (free tier at [tavily.com](https://tavily.com)) — required for `MedicalWebSearchTool`

### 📈 3. Build the SQLite databases

```bash
python src/prepare_db.py
```

This creates `heart_disease.db`, `cancer.db`, and `diabetes.db` inside
`data/db/` with correctly typed columns (INTEGER / REAL / TEXT).

### 4. Run the agent

```bash
python main.py
```

## 💬 Sample Questions

**📊 Statistics (routed to a DB tool):**
- "What is the average cholesterol level in the heart disease dataset?"
- "What is the average single epithelial cell size in the cancer data?"
- "What is the average age of patients diagnosed with diabetes?"

**📖 General medical knowledge (routed to MedicalWebSearchTool):**
- "What are the common symptoms of heart disease?"
- "What causes type 2 diabetes?"
- "How is breast cancer typically treated?"

**🔀 Mixed (routed to both):**
- "What is diabetes, and what is the average glucose level in the dataset?"

## 🔒 Safety Notes

- All three DB tools connect in **read-only** query mode and explicitly
  block write operations (`INSERT` / `UPDATE` / `DELETE` / `DROP` /
  `ALTER`) so the agent can never modify the underlying data.
- API keys are loaded from a local `.env` file (never committed —
  see `.gitignore`).

## 🛠️ Key Frameworks Used

| Tool | Purpose |
|---|---|
| **LangChain** | `create_agent` (LangGraph-based tool calling) |
| **Google Gemini** | Primary LLM backend via `langchain-google-genai` |
| **OpenAI-compatible function calling** | Fallback via Azure OpenAI or the OpenAI API |
| **SQLite** | Lightweight structured storage for all three datasets |
| **Tavily Search API** | General medical knowledge retrieval |

## 🖊️ Author

**Farjana Ferdausi**

AI/ML Engineering & Data Science, Fellow — Google Cloud Gen AI Academy APAC Edition (Cohort 3) | Agentic AI · RAG · Gemini · ADK · BigQuery MCP · Cloud Run | Former HR Professional (14+ years) at Radisson Blu Dhaka Water Garden, Bangladesh

Built for as part of the AI/ML Engineering program | Batch : 6 |
Module : 23 | Ostad | Bangladesh |

[GitHub](https://github.com/farjanaferdausi-cs50ai) · [LinkedIn](https://linkedin.com/in/farjana-ferdausi/) · [Medium](https://medium.com/@farjana.rafi1983)


## ⚠️ A note on Gemini's "thinking" models

I originally built this with LangChain's older `create_tool_calling_agent`
+ `AgentExecutor` combo. Gemini's newer 3.x "thinking" models attach an
internal `thought_signature` to every tool call, which must be echoed
back to the API on the next turn — the old `AgentExecutor` rebuilds its
message scratchpad and drops that signature, causing a
`Function call is missing a thought_signature` error. I moved to
LangChain's modern `create_agent` (built on LangGraph), which keeps the
original message objects intact across turns, so the signature survives
automatically. This is also why `requirements.txt` no longer pins old
0.3.x LangChain versions — the modern stack is required for Gemini 3.x
tool calling to work correctly.
