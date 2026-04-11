"""
Dense retrieval using Contriever (facebook/contriever).
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

# 全局模型（懒加载，只加载一次）
_dense_model = None


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        # 使用 facebook/contriever-msmarco 或 facebook/contriever
        _dense_model = SentenceTransformer("facebook/contriever")
    return _dense_model


def dense_retrieve(
    query: str,
    docs: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Dense retrieval using Contriever.

    Args:
        query: question string.
        docs: list of {"title": str, "text": str, ...}
        top_k: number of documents to return.

    Returns:
        list[dict]: top-k documents sorted by dense similarity (highest first).
        Each dict is a copy of the original doc with added keys:
            - "dense_score": float (cosine similarity)
            - "original_index": int (0-based index in the input list)
    """
    if not docs:
        return []

    model = _get_dense_model()

    # 1. 构建所有文档的文本表示（title + text）
    doc_texts = [f"{d['title']} {d['text']}" for d in docs]

    # 2. 编码 query 和所有文档（一次性批量编码）
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(doc_texts, convert_to_tensor=True)

    # 3. 计算余弦相似度（query vs 每个文档）
    similarities = util.cos_sim(query_emb, doc_embs)[0]  # shape: (len(docs),)

    # 4. 转换为 numpy 并获取 top-k 索引
    sim_np = similarities.cpu().numpy()
    top_indices = np.argsort(sim_np)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc = dict(docs[idx])  # 浅拷贝，保留原始字段
        doc["dense_score"] = float(sim_np[idx])
        doc["original_index"] = int(idx)
        results.append(doc)

    return results