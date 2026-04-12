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
    "You MUST use the following output format:\n\n"
    "Answer: <short factual answer, a few words or one sentence>\n"
    "Citations: [Doc N] for each document that supports your answer\n\n"
    "If the answer cannot be determined from the documents, write:\n"
    "Answer: I cannot determine the answer from the provided documents.\n"
    "Citations: none"
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
        f"Respond using the exact format: Answer: ... then Citations: ..."
    )


def build_no_rag_user_prompt(question: str) -> str:
    """
    Build the user message for the No-RAG baseline (E0).
    """
    return f"Question: {question}\n\nAnswer concisely."


# ====================== 追加到 prompts.py 末尾 ======================
# ------------------------------ E7 Self-RAG Prompts ------------------------------
SELF_RAG_CRITIQUE_SYSTEM_PROMPT = (
    "You are a factual critique assistant. Your task is to evaluate whether the given answer "
    "is strictly supported by the provided documents, and identify any hallucinations, unsupported claims, "
    "or citation errors. You must output your judgment in the following format:\n\n"
    "HasHallucination: YES/NO\n"
    "Critique: <brief explanation of the issue, or 'No issues' if the answer is fully supported>\n"
    "Suggestion: <rewrite suggestion to fix the issue, or 'None' if no issues>\n\n"
    "Rules:\n"
    "1. Only mark HasHallucination as YES if there is content not supported by the documents.\n"
    "2. Ignore minor wording differences, only focus on factual correctness.\n"
    "3. If the answer says 'cannot determine', mark HasHallucination as NO."
)

SELF_RAG_REGEN_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Rewrite the answer to fix the hallucination issues mentioned in the critique, "
    "using ONLY the information in the provided documents. "
    "You MUST use the following output format:\n\n"
    "Answer: <short factual answer, a few words or one sentence>\n"
    "Citations: [Doc N] for each document that supports your answer\n\n"
    "If the answer cannot be determined from the documents, write:\n"
    "Answer: I cannot determine the answer from the provided documents.\n"
    "Citations: none"
)

def build_critique_prompt(question: str, answer: str, docs: list[dict]) -> str:
    """Build user prompt for Self-RAG critique step."""
    doc_str = format_docs(docs)
    return (
        f"Documents:\n{doc_str}\n\n"
        f"Question: {question}\n"
        f"Answer to evaluate: {answer}\n\n"
        f"Evaluate the answer strictly against the documents, output using the required format."
    )

def build_regeneration_prompt(question: str, docs: list[dict], original_answer: str, critique: str) -> str:
    """Build user prompt for Self-RAG regeneration step."""
    doc_str = format_docs(docs)
    return (
        f"Documents:\n{doc_str}\n\n"
        f"Question: {question}\n"
        f"Original answer: {original_answer}\n"
        f"Critique: {critique}\n\n"
        f"Rewrite the answer to fix the issues, using only the provided documents, and follow the required format."
    )

# ------------------------------ E8 Retrieval Enhancement Prompts ------------------------------
QUERY_REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriting assistant for open-domain multi-hop question answering. "
    "Rewrite the user's question into 2-3 optimized search queries that better retrieve relevant documents. "
    "Rules:\n"
    "1. Break down multi-hop questions into clear sub-queries.\n"
    "2. Preserve all core factual entities and key information.\n"
    "3. Output only the rewritten queries, one per line, no extra explanation."
)

def build_query_rewrite_prompt(question: str) -> str:
    """Build user prompt for query rewriting."""
    return f"Original question: {question}\n\nRewrite into optimized search queries:"