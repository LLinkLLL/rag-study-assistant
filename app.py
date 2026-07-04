import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.app_errors import format_exception_for_user
from src.chunker import chunk_pages
from src.config import get_chat_model, get_min_relevance_score
from src.feedback_chain import generate_feedback
from src.pdf_loader import extract_pdf_pages
from src.qa_chain import answer_question
from src.quiz_chain import generate_quiz
from src.retriever import filter_documents_by_relevance, retrieve_documents_with_scores
from src.summary_chain import summarize_documents, summarize_full_document
from src.vector_store import (
    add_documents_to_store,
    clear_vector_store,
    count_documents,
    get_all_documents,
    get_vector_store,
)


load_dotenv()

APP_TITLE = "RAG-based Study Assistant with Source-grounded Answer Feedback"
DEFAULT_PERSIST_DIR = "chroma_db"


st.set_page_config(
    page_title="RAG Study Assistant",
    layout="wide",
)


def require_api_key() -> bool:
    """Show a friendly warning if the OpenAI API key is missing."""
    if os.getenv("OPENAI_API_KEY"):
        return True

    st.warning(
        "Add your OpenAI API key to a `.env` file before generating answers. "
        "See `.env.example` for the expected format."
    )
    return False


def show_frontend_error(error: Exception, context: str) -> None:
    """Render a safe, user-friendly error message in the Streamlit UI."""
    user_error = format_exception_for_user(error, context)
    st.error(user_error.title)
    st.write(user_error.message)
    st.info(user_error.suggestion)
    with st.expander("Technical details"):
        st.code(user_error.technical_detail)


def render_sources(documents, title: str = "Sources", min_relevance_score: float | None = None) -> None:
    """Display retrieved source chunks with their metadata."""
    if not documents:
        st.info("No source chunks were retrieved.")
        return

    st.subheader(title)
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        filename = metadata.get("filename", "Unknown file")
        page_number = metadata.get("page_number", "Unknown page")
        chunk_id = metadata.get("chunk_id", "Unknown chunk")
        relevance_score = metadata.get("relevance_score")
        score_label = f" | score {relevance_score:.2f}" if isinstance(relevance_score, float) else ""
        threshold_label = ""
        if isinstance(relevance_score, float) and min_relevance_score is not None:
            threshold_label = " | relevant" if relevance_score >= min_relevance_score else " | below threshold"
        label = f"{index}. {filename} | page {page_number} | chunk {chunk_id}{score_label}{threshold_label}"

        with st.expander(label):
            st.write(doc.page_content)


@st.cache_resource(show_spinner=False)
def load_vector_store(persist_directory: str):
    """Keep one Chroma connection alive across Streamlit reruns."""
    return get_vector_store(persist_directory=persist_directory)


def main() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Upload course PDFs, index them in ChromaDB, and generate answers, summaries, "
        "quizzes, and feedback that stay grounded in the uploaded material."
    )

    with st.sidebar:
        st.header("Settings")
        persist_directory = st.text_input("ChromaDB directory", DEFAULT_PERSIST_DIR)
        model_name = st.text_input("OpenAI chat model", get_chat_model())
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=4)
        min_relevance_score = st.slider(
            "Minimum relevance score",
            min_value=0.0,
            max_value=1.0,
            value=get_min_relevance_score(),
            step=0.05,
            help="Chunks below this score are treated as insufficient evidence.",
        )
        st.divider()

        vector_store = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                vector_store = load_vector_store(persist_directory)
                indexed_count = count_documents(vector_store)
                st.metric("Indexed chunks", indexed_count)
            except Exception as error:
                st.metric("Indexed chunks", "-")
                show_frontend_error(error, "Loading vector database")
        else:
            st.metric("Indexed chunks", "-")
            st.info("Add `OPENAI_API_KEY` to `.env` to enable indexing and retrieval.")

        if st.button("Reset document database", type="secondary"):
            try:
                clear_vector_store(persist_directory, vector_store=vector_store)
                load_vector_store.clear()
                st.success("The ChromaDB index was cleared.")
                st.rerun()
            except Exception as error:
                show_frontend_error(error, "Resetting document database")

    upload_tab, ask_tab, summary_tab, quiz_feedback_tab = st.tabs(
        ["Upload & Index", "Ask Questions", "Summarize", "Quiz & Feedback"]
    )

    with upload_tab:
        st.header("Upload & Index")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF course files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            chunk_size = st.number_input("Chunk size", min_value=300, max_value=3000, value=1000, step=100)
        with col_b:
            chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=1000,
                value=200,
                step=50,
            )

        if st.button("Extract and index PDFs", type="primary", disabled=not uploaded_files):
            if not require_api_key():
                st.stop()

            try:
                if vector_store is None:
                    vector_store = load_vector_store(persist_directory)

                all_pages = []
                with st.spinner("Extracting text from PDFs..."):
                    for uploaded_file in uploaded_files:
                        pages = extract_pdf_pages(uploaded_file.getvalue(), uploaded_file.name)
                        all_pages.extend(pages)

                if not all_pages:
                    st.error("No extractable text was found in the uploaded PDFs.")
                    st.stop()

                with st.spinner("Splitting text and storing chunks in ChromaDB..."):
                    documents = chunk_pages(
                        all_pages,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    added_ids = add_documents_to_store(vector_store, documents)
            except Exception as error:
                show_frontend_error(error, "Indexing uploaded PDFs")
                st.stop()

            skipped_count = len(documents) - len(added_ids)
            st.success(
                f"Indexed {len(added_ids)} new chunks from {len(uploaded_files)} PDF file(s). "
                f"Skipped {skipped_count} duplicate chunk(s)."
            )

            preview_rows = [
                {
                    "filename": doc.metadata.get("filename"),
                    "page": doc.metadata.get("page_number"),
                    "chunk": doc.metadata.get("chunk_id"),
                    "characters": len(doc.page_content),
                }
                for doc in documents[:20]
            ]
            st.dataframe(preview_rows, use_container_width=True)

    with ask_tab:
        st.header("Ask Questions")
        question = st.text_area(
            "Question",
            placeholder="Example: What are the main assumptions behind this algorithm?",
        )

        if st.button("Answer question", type="primary"):
            if not require_api_key():
                st.stop()
            if not question.strip():
                st.warning("Enter a question first.")
                st.stop()

            try:
                if vector_store is None:
                    vector_store = load_vector_store(persist_directory)

                with st.spinner("Retrieving relevant PDF chunks..."):
                    all_retrieved_docs = retrieve_documents_with_scores(
                        vector_store,
                        question,
                        k=top_k,
                    )
                    retrieved_docs = filter_documents_by_relevance(
                        all_retrieved_docs,
                        min_relevance_score=min_relevance_score,
                    )

                with st.spinner("Generating a source-grounded answer..."):
                    answer = answer_question(
                        question,
                        all_retrieved_docs,
                        model_name=model_name,
                        min_relevance_score=min_relevance_score,
                    )
            except Exception as error:
                show_frontend_error(error, "Answering question")
                st.stop()

            st.subheader("Answer")
            st.markdown(answer)
            if all_retrieved_docs and not retrieved_docs:
                st.warning(
                    "Chunks were retrieved, but all of them were below the current relevance threshold. "
                    "The answer was still generated from the displayed snippets, so treat it as limited evidence."
                )
            render_sources(all_retrieved_docs, "Retrieved Sources and Scores", min_relevance_score)

    with summary_tab:
        st.header("Summarize")
        summary_mode = st.radio(
            "Summary mode",
            ["Topic-based retrieval summary", "Full-document map-reduce summary"],
            horizontal=True,
        )
        summary_topic = st.text_input(
            "Optional focus topic",
            placeholder="Example: gradient descent, database normalization, lecture 3",
        )
        summary_style = st.selectbox(
            "Summary style",
            ["Study notes", "Concise overview", "Exam revision bullets"],
        )
        max_summary_chunks = st.slider("Maximum chunks to summarize", 3, 30, 12)

        if st.button("Generate summary", type="primary"):
            if not require_api_key():
                st.stop()

            try:
                if vector_store is None:
                    vector_store = load_vector_store(persist_directory)

                all_summary_docs = []
                with st.spinner("Collecting source chunks..."):
                    if summary_mode == "Full-document map-reduce summary":
                        summary_docs = get_all_documents(vector_store, limit=None)
                    elif summary_topic.strip():
                        all_summary_docs = retrieve_documents_with_scores(
                            vector_store,
                            summary_topic,
                            k=max_summary_chunks,
                        )
                        summary_docs = filter_documents_by_relevance(
                            all_summary_docs,
                            min_relevance_score=min_relevance_score,
                        )
                    else:
                        summary_docs = get_all_documents(vector_store, limit=max_summary_chunks)

                with st.spinner("Generating summary..."):
                    if summary_mode == "Full-document map-reduce summary":
                        summary = summarize_full_document(
                            summary_docs,
                            style=summary_style,
                            model_name=model_name,
                        )
                    else:
                        summary = summarize_documents(
                            summary_docs,
                            topic=summary_topic,
                            style=summary_style,
                            model_name=model_name,
                        )
            except Exception as error:
                show_frontend_error(error, "Generating summary")
                st.stop()

            st.subheader("Summary")
            st.markdown(summary)
            if summary_mode == "Topic-based retrieval summary" and summary_topic.strip():
                render_sources(all_summary_docs, "Summary Retrieval Scores", min_relevance_score)
            else:
                render_sources(summary_docs, "Summary Sources")

    with quiz_feedback_tab:
        st.header("Quiz & Feedback")
        quiz_col, feedback_col = st.columns(2)

        with quiz_col:
            st.subheader("Generate Quiz")
            quiz_topic = st.text_input("Quiz topic", placeholder="Example: neural networks")
            quiz_count = st.slider("Number of questions", 3, 10, 5)
            quiz_difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])

            if st.button("Generate quiz", type="primary"):
                if not require_api_key():
                    st.stop()
                if not quiz_topic.strip():
                    st.warning("Enter a quiz topic first.")
                    st.stop()

                try:
                    if vector_store is None:
                        vector_store = load_vector_store(persist_directory)

                    with st.spinner("Retrieving topic sources..."):
                        all_quiz_docs = retrieve_documents_with_scores(
                            vector_store,
                            quiz_topic,
                            k=top_k,
                        )
                        quiz_docs = filter_documents_by_relevance(
                            all_quiz_docs,
                            min_relevance_score=min_relevance_score,
                        )

                    with st.spinner("Writing quiz questions..."):
                        quiz = generate_quiz(
                            quiz_topic,
                            quiz_docs,
                            question_count=quiz_count,
                            difficulty=quiz_difficulty,
                            model_name=model_name,
                        )
                except Exception as error:
                    show_frontend_error(error, "Generating quiz")
                    st.stop()

                st.markdown(quiz)
                if all_quiz_docs and not quiz_docs:
                    st.warning("Retrieved quiz sources were all below the current relevance threshold.")
                render_sources(all_quiz_docs, "Quiz Retrieval Scores", min_relevance_score)

        with feedback_col:
            st.subheader("Student Answer Feedback")
            feedback_question = st.text_area(
                "Question or task",
                placeholder="Paste the original question the student answered.",
            )
            student_answer = st.text_area(
                "Student answer",
                placeholder="Paste the student's answer here.",
                height=160,
            )
            feedback_topic = st.text_input(
                "Optional feedback topic",
                placeholder="Example: photosynthesis, recursion, market segmentation",
            )

            if st.button("Generate feedback", type="primary"):
                if not require_api_key():
                    st.stop()
                if not feedback_question.strip() or not student_answer.strip():
                    st.warning("Enter both the question and the student's answer.")
                    st.stop()

                retrieval_query = " ".join(
                    part.strip()
                    for part in [feedback_topic, feedback_question, student_answer]
                    if part.strip()
                )

                try:
                    if vector_store is None:
                        vector_store = load_vector_store(persist_directory)

                    with st.spinner("Retrieving relevant source material..."):
                        all_feedback_docs = retrieve_documents_with_scores(
                            vector_store,
                            retrieval_query,
                            k=top_k,
                        )
                        feedback_docs = filter_documents_by_relevance(
                            all_feedback_docs,
                            min_relevance_score=min_relevance_score,
                        )

                    with st.spinner("Generating grounded feedback..."):
                        feedback = generate_feedback(
                            question=feedback_question,
                            student_answer=student_answer,
                            documents=feedback_docs,
                            model_name=model_name,
                        )
                except Exception as error:
                    show_frontend_error(error, "Generating feedback")
                    st.stop()

                st.markdown(feedback)
                if all_feedback_docs and not feedback_docs:
                    st.warning("Retrieved feedback sources were all below the current relevance threshold.")
                render_sources(all_feedback_docs, "Feedback Retrieval Scores", min_relevance_score)


if __name__ == "__main__":
    Path(DEFAULT_PERSIST_DIR).mkdir(exist_ok=True)
    main()
