"""Citation parsing and grounding evaluation."""

import re


_CITE_PATTERN = re.compile(r"\[Doc\s*(\d+)\]")


def extract_citations(text: str) -> list[int]:
    """
    Extract cited document indices (1-based) from generated text.
    """
    return [int(m) for m in _CITE_PATTERN.findall(text)]


def citation_grounding_rate(
    generated_text: str,
    docs: list[dict],
    gold_titles: set[str],
) -> dict:
    """
    Measure how well citations align with gold supporting documents.

    Returns:
        dict with keys: cited_indices, num_citations, num_grounded,
                        grounding_rate
    """
    cited = extract_citations(generated_text)
    unique_cited = sorted(set(cited))

    num_grounded = 0
    for idx in unique_cited:
        if 1 <= idx <= len(docs):
            doc_title = docs[idx - 1]["title"]
            if doc_title in gold_titles:
                num_grounded += 1

    n = len(unique_cited) if unique_cited else 1  # avoid div-by-zero
    return {
        "cited_indices": unique_cited,
        "num_citations": len(unique_cited),
        "num_grounded": num_grounded,
        "grounding_rate": num_grounded / n,
    }
