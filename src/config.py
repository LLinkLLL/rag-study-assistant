from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()

DEFAULT_CHAT_MODEL = "gpt-5.4-nano"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MIN_RELEVANCE_SCORE = 0.2


def get_chat_model() -> str:
    """Return the configured OpenAI chat model with a reliable fallback."""
    return os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL


def get_embedding_model() -> str:
    """Return the configured OpenAI embedding model with a reliable fallback."""
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_min_relevance_score() -> float:
    """Read the relevance threshold from .env and keep it inside [0, 1]."""
    raw_value = os.getenv("MIN_RELEVANCE_SCORE", str(DEFAULT_MIN_RELEVANCE_SCORE))
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_MIN_RELEVANCE_SCORE

    return max(0.0, min(1.0, value))
