from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_chat_model
from src.prompts import FEEDBACK_PROMPT
from src.qa_chain import format_documents_for_prompt, has_enough_context


def generate_feedback(
    question: str,
    student_answer: str,
    documents: list[Document],
    model_name: str | None = None,
) -> str:
    """Generate rubric-style feedback grounded in retrieved source chunks."""
    if not has_enough_context(documents):
        return "There is not enough information in the uploaded materials to grade this answer reliably."

    prompt = ChatPromptTemplate.from_template(FEEDBACK_PROMPT)
    llm = ChatOpenAI(model=model_name or get_chat_model(), temperature=0)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "question": question,
            "student_answer": student_answer,
            "context": format_documents_for_prompt(documents),
        }
    )
