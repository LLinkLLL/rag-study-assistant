from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_chat_model
from src.prompts import QUIZ_PROMPT
from src.qa_chain import format_documents_for_prompt, has_enough_context


def generate_quiz(
    topic: str,
    documents: list[Document],
    question_count: int = 5,
    difficulty: str = "Intermediate",
    model_name: str | None = None,
) -> str:
    """Generate quiz questions based on retrieved source material."""
    if not has_enough_context(documents):
        return "I cannot find enough information in the uploaded materials to create a quiz on that topic."

    prompt = ChatPromptTemplate.from_template(QUIZ_PROMPT)
    llm = ChatOpenAI(model=model_name or get_chat_model(), temperature=0.4)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "topic": topic,
            "difficulty": difficulty,
            "question_count": question_count,
            "context": format_documents_for_prompt(documents),
        }
    )
