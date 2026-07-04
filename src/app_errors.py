from __future__ import annotations

import re
from dataclasses import dataclass


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{12,}"),
]


@dataclass(frozen=True)
class UserFacingError:
    """A concise error message that is safe to show in the Streamlit UI."""

    title: str
    message: str
    suggestion: str
    technical_detail: str


def sanitize_error_text(text: str) -> str:
    """Remove obvious secrets from error text before displaying it."""
    sanitized = text or "No technical details were provided."
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[redacted-api-key]", sanitized)

    return sanitized


def format_exception_for_user(error: Exception, context: str) -> UserFacingError:
    """Convert common API/model/config errors into helpful UI copy."""
    error_type = error.__class__.__name__
    raw_message = sanitize_error_text(str(error))
    normalized = f"{error_type} {raw_message}".lower()

    if any(term in normalized for term in ["authentication", "api key", "unauthorized", "401"]):
        return UserFacingError(
            title=f"{context}: OpenAI API key problem",
            message="The app could not authenticate with OpenAI.",
            suggestion="Check `OPENAI_API_KEY` in `.env`, then restart Streamlit.",
            technical_detail=f"{error_type}: {raw_message}",
        )

    if any(term in normalized for term in ["model", "does not exist", "not found", "invalid_request", "404"]):
        return UserFacingError(
            title=f"{context}: OpenAI model or request problem",
            message="The selected OpenAI model or request configuration was rejected.",
            suggestion="Check `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` in `.env`, and confirm your account can use them.",
            technical_detail=f"{error_type}: {raw_message}",
        )

    if any(term in normalized for term in ["rate limit", "quota", "insufficient_quota", "429"]):
        return UserFacingError(
            title=f"{context}: OpenAI quota or rate limit problem",
            message="OpenAI rejected the request because of quota or rate limits.",
            suggestion="Wait and retry, or check billing and usage limits in your OpenAI account.",
            technical_detail=f"{error_type}: {raw_message}",
        )

    if any(term in normalized for term in ["connection", "timeout", "network", "dns", "ssl"]):
        return UserFacingError(
            title=f"{context}: Network connection problem",
            message="The app could not reach the API service reliably.",
            suggestion="Check your internet connection, proxy/VPN settings, and retry.",
            technical_detail=f"{error_type}: {raw_message}",
        )

    if any(term in normalized for term in ["chroma", "sqlite", "database", "collection"]):
        return UserFacingError(
            title=f"{context}: Vector database problem",
            message="The local ChromaDB database could not be read or written.",
            suggestion="Try the `Reset document database` button. If that fails, stop Streamlit and delete `chroma_db/`.",
            technical_detail=f"{error_type}: {raw_message}",
        )

    return UserFacingError(
        title=f"{context}: Unexpected error",
        message="Something went wrong while processing this request.",
        suggestion="Review the technical detail below, then check your `.env` settings and uploaded document.",
        technical_detail=f"{error_type}: {raw_message}",
    )
