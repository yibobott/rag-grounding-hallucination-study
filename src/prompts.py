"""
Prompt templates for RAG experiments.
"""


def format_docs(docs: list[dict], start_index: int = 1) -> str:
    """
    Format a list of documents into a numbered string.

    Args:
        docs: list of {"title": str, "text": str}
        start_index: numbering starts from this value (default 1)

    Returns:
        Formatted string, e.g.:
        [Doc 1] Title: Albert Einstein
        Albert Einstein was a German-born theoretical physicist ...
    """
    parts = []
    for i, doc in enumerate(docs, start=start_index):
        parts.append(f"[Doc {i}] Title: {doc['title']}\n{doc['text']}")
    return "\n\n".join(parts)


# ------------------------------ System prompts ------------------------------ #

RAG_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the information in the provided documents. "
    "Be concise — give a short factual answer (a few words or one sentence). "
    "Cite your sources using [Doc N] tags after the relevant part of your answer. "
    "If the answer cannot be determined from the documents, say \"I cannot determine the answer from the provided documents.\""
)

NO_RAG_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question concisely — give a short factual answer (a few words or one sentence). "
    "Do not make up information. If you are unsure, say so."
)


# --------------------------- User prompt builders --------------------------- #

def build_rag_user_prompt(question: str, docs: list[dict]) -> str:
    """
    Build the user message for a RAG setting (E-Oracle, E1–E3, etc.).
    """
    doc_str = format_docs(docs)
    return (
        f"Documents:\n{doc_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely and cite relevant documents using [Doc N] tags."
    )


def build_no_rag_user_prompt(question: str) -> str:
    """
    Build the user message for the No-RAG baseline (E0).
    """
    return f"Question: {question}\n\nAnswer concisely."
