"""
config.py
---------
I keep all environment loading and LLM initialization here so every
other module simply imports `get_llm()` instead of repeating the
setup code.

I support FOUR possible LLM backends, checked in this priority order,
so this project works no matter which credentials I have available:

    1. Google Gemini      (GOOGLE_API_KEY, from Google AI Studio —
                            my primary option, since GitHub Models was
                            permanently retired on July 30, 2026)
    2. Azure OpenAI       (matches my test_OpenAI_models.ipynb setup)
    3. Direct OpenAI API  (OPENAI_API_KEY)
    4. GitHub Models      (kept for completeness, but this service was
                            retired by GitHub on July 30, 2026 and will
                            raise an error if used)

I only need to fill in ONE of these in my .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Option 1: Google Gemini (primary, free at aistudio.google.com) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-3.5-flash-lite")

# --- Option 2: Azure OpenAI ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Option 3: Direct OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Option 4: GitHub Models (RETIRED July 30, 2026 — kept for reference only) ---
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
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
        )

            # I disable "thinking" mode here on purpose. Gemini's newer
            # (3.x) thinking models attach an internal "thought_signature"
            # to every function/tool call, and my agent framework
            # (LangChain's classic AgentExecutor) does not yet round-trip
            # that signature back to the API, which causes a
            # "Function call is missing a thought_signature" 400 error on
            # any multi-step tool call. Setting thinking_budget=0 avoids
            # the whole issue since the model no longer produces
            # thought signatures in the first

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
        # NOTE: GitHub Models was permanently retired on July 30, 2026.
        # This branch is kept only for historical reference and will
        # fail with an APIStatusError if actually used.
        return ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=GITHUB_TOKEN,
            openai_api_base=MODEL_ENDPOINT,
            temperature=temperature,
        )

    raise ValueError(
        "No LLM credentials found in .env. Please fill in ONE of: "
        "GOOGLE_API_KEY (recommended), Azure OpenAI settings, or "
        "OPENAI_API_KEY. See .env.example for the exact variable names."
    )


