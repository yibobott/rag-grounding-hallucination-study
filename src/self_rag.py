# src/self_rag.py (最终适配版)
import re
from src.generation import generate
from src.retrieval import hybrid_retrieve
from src.evaluation.factscore import (
    rewrite_as_statement,
    decompose_into_claims,
    verify_claim
)

def self_rag_generate(
    question: str,
    model_key: str,
    top_k: int = 5,
    rrf_k: int = 60,
    threshold: float = 0.7
) -> str:
    """
    Self-RAG 完整生成流水线（完全适配你项目的 factscore 函数）
    1. 初始答案生成 → 2. 改写为陈述 → 3. 拆分为声明 → 4. 逐一验证 → 5. 答案修正
    """
    # --------------------------
    # Step1: 初始答案生成
    # --------------------------
    initial_docs = hybrid_retrieve(question, top_k=top_k, rrf_k=rrf_k)
    
    initial_prompt = f"""
    You are a question-answering assistant. Answer the user's question using ONLY the provided reference documents.
    Cite the documents you use with [Doc N] tags (e.g., [Doc 1]).
    If you cannot answer the question based on the documents, say "I cannot answer this question based on the provided documents."
    Do NOT fabricate any information.

    Reference documents:
    {''.join([f'[Doc {i+1}] {doc["title"]}: {doc["text"]}\n' for i, doc in enumerate(initial_docs)])}

    Question: {question}
    Answer:
    """
    
    base_model_key = model_key.replace("-self-rag", "")
    initial_answer = generate(
        model_key=base_model_key,
        prompt=initial_prompt,
        max_tokens=512,
        temperature=0.0
    )

    # --------------------------
    # Step2-4: 用你项目的 factscore 流水线做事实校验
    # --------------------------
    try:
        # Step2: 改写为陈述
        statement = rewrite_as_statement(question, initial_answer, base_model_key)
        # Step3: 拆分为声明
        claims = decompose_into_claims(statement, base_model_key)
        
        if not claims:
            return initial_answer
            
        # Step4: 逐一验证声明
        supported_claims = []
        unsupported_claims = []
        
        for claim in claims:
            # 针对单个声明重新检索
            claim_docs = hybrid_retrieve(claim, top_k=3, rrf_k=rrf_k)
            # 用你项目的 verify_claim 函数验证
            is_supported = verify_claim(claim, claim_docs, base_model_key)
            
            if is_supported:
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
                
    except Exception as e:
        # 如果 factscore 流水线报错，直接返回初始答案
        print(f"Self-RAG 校验跳过: {e}")
        return initial_answer

    # --------------------------
    # Step5: 修正答案
    # --------------------------
    if not unsupported_claims:
        return initial_answer
    
    # 基于支持的声明重写答案
    rewrite_prompt = f"""
    You are a precise answer editor. Rewrite the initial answer to ONLY keep information that is fully supported.
    Remove all unsupported claims, do NOT add any new information, keep the original citation format.

    Original question: {question}
    Reference documents:
    {''.join([f'[Doc {i+1}] {doc["title"]}: {doc["text"]}\n' for i, doc in enumerate(initial_docs)])}
    Initial answer: {initial_answer}
    Supported claims: {'; '.join(supported_claims)}
    Unsupported claims to remove: {'; '.join(unsupported_claims)}

    Final revised answer:
    """
    
    final_answer = generate(
        model_key=base_model_key,
        prompt=rewrite_prompt,
        max_tokens=512,
        temperature=0.0
    )
    
    return final_answer