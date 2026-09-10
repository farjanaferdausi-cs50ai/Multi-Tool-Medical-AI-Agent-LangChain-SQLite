"""
config.py
---------
I keep all environment loading and LLM initialization here so every
other module simply imports `get_llm()` instead of repeating the
setup code.

I support THREE possible LLM backends, checked in this priority order,
so this project works no matter which credentials I have available:

    1. Azure OpenAI       (matches my test_OpenAI_models.ipynb setup)
    2. Direct OpenAI API  (OPENAI_API_KEY)
    3. GitHub Models      (matches LangChain_Tools_and_Resilience.ipynb,
                            free with a GitHub account)

I only need to fill in ONE of these three in my .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Option 1: Azure OpenAI ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Option 2: Direct OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Option 3: GitHub Models (free fallback) ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "https://models.github.ai/inference")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

DB_DIR = BASE_DIR / "data" / "db"
HEART_DB_PATH = DB_DIR / "heart_disease.db"
CANCER_DB_PATH = DB_DIR / "cancer.db"
DIABETES_DB_PATH = DB_DIR / "diabetes.db"


def get_llm(temperature: float = 0.0):
    """
    I centralize LLM creation here so agent.py and every tool file just
    call get_llm() without caring which backend is actually configured.
    """
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT_NAME:
        return AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=temperature,
        )

    if OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=temperature)

    if GITHUB_TOKEN:
        return ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=GITHUB_TOKEN,
            openai_api_base=MODEL_ENDPOINT,
            temperature=temperature,
        )

    raise ValueError(
        "No LLM credentials found in .env. Please fill in ONE of: "
        "Azure OpenAI settings, OPENAI_API_KEY, or GITHUB_TOKEN. "
        "See .env.example for the exact variable names."
    )

