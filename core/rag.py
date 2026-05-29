"""
rag.py
RAG 문서 검색 모듈

포함 함수:
- load_embedding_model() : KURE-v1 임베딩 모델 로드
- load_documents()       : JSON 문서 + 임베딩 벡터 로드
- embed_query()          : 쿼리 텍스트 → 임베딩 벡터
- retrieve_documents()   : 단일 쿼리로 유사 문서 검색
- retrieve_with_fusion() : 쿼리 그룹별 검색 + score fusion + 중복 제거
"""

import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


EMBED_MODEL_NAME = "nlpai-lab/KURE-v1"
QUERY_PREFIX     = "query: "
DOCUMENT_PREFIX  = "passage: "


# ============================================================
# 모델 & 문서 로드
# ============================================================

def load_embedding_model(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """KURE-v1 임베딩 모델 로드 (GPU 있으면 자동 사용)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(model_name, device=device)
    print(f"임베딩 모델 로드 완료 | 모델: {model_name} | device: {device}")
    return model


def load_documents(json_path: str) -> tuple[list, np.ndarray, np.ndarray]:
    """
    임베딩된 JSON 문서 로드
    Args:
        json_path: baemin_articles_embedded.json 경로
    Returns:
        docs                  : 문서 리스트
        doc_title_embeddings  : (N, 768) title 임베딩 행렬
        doc_body_embeddings   : (N, 768) body 임베딩 행렬
    """
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    doc_title_embeddings = np.array([doc["title_embedding"] for doc in docs])
    doc_body_embeddings  = np.array([doc["body_embedding"]  for doc in docs])

    print(f"문서 로드 완료 | 총 {len(docs)}개 | shape: {doc_title_embeddings.shape}")
    return docs, doc_title_embeddings, doc_body_embeddings


# ============================================================
# 임베딩 & 검색
# ============================================================

def embed_query(query: str, embedding_model: SentenceTransformer) -> np.ndarray:
    """
    쿼리 텍스트를 임베딩 벡터로 변환
    Returns:
        shape (1, 768) 임베딩 벡터
    """
    emb = embedding_model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    return emb.reshape(1, -1)


def retrieve_documents(
    query: str,
    docs: list,
    doc_title_embeddings: np.ndarray,
    doc_body_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    top_k: int = 3,
    title_weight: float = 0.3,
    body_weight: float = 0.7,
) -> list:
    """
    단일 쿼리로 유사 문서 top_k개 검색
    Returns:
        [
            {
                "score"      : 최종 유사도 (0~1),
                "title_score": 제목 유사도,
                "body_score" : 본문 유사도,
                "title"      : 문서 제목,
                "category_1" : 대분류,
                "category_2" : 소분류,
                "raw_text"   : 원문 전체,
                "text"       : 원문 앞 500자,
            },
            ...
        ]
    """
    query_embedding = embed_query(query, embedding_model)
    title_sims      = cosine_similarity(query_embedding, doc_title_embeddings)[0]
    body_sims       = cosine_similarity(query_embedding, doc_body_embeddings)[0]
    combined_sims   = title_weight * title_sims + body_weight * body_sims

    top_indices = np.argsort(combined_sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score":       float(combined_sims[idx]),
            "title_score": float(title_sims[idx]),
            "body_score":  float(body_sims[idx]),
            "title":       docs[idx]["title"],
            "category_1":  docs[idx]["category_1"],
            "category_2":  docs[idx]["category_2"],
            "raw_text":    docs[idx]["raw_text"],
            "text":        docs[idx]["raw_text"][:500],
        })

    return results


def retrieve_with_fusion(
    query_groups: dict,
    docs: list,
    doc_title_embeddings: np.ndarray,
    doc_body_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    top_k_per_query: int = 3,
    final_top_k: int = 5,
    group_weights: dict = None,
) -> pd.DataFrame:
    """
    쿼리 그룹별 검색 후 score fusion으로 최종 문서 선택
    - 같은 문서가 여러 쿼리에서 검색되면 점수 누적합산 (중복 제거 아님)
    - group_weights로 쿼리 그룹별 가중치 조정 가능

    Args:
        query_groups   : {"정형": [...], "리뷰": [...], "사장님": [...]}
        group_weights  : {"정형": 1.0, "리뷰": 1.0}  (미지정 시 1.0)
        top_k_per_query: 쿼리당 검색 문서 수
        final_top_k    : 최종 반환 문서 수

    Returns:
        DataFrame (final_score, hit_count, matched_queries 컬럼 포함)
    """
    if group_weights is None:
        group_weights = {}

    score_map = {}  # {doc_title: {"score": float, "hits": int, "queries": [], "doc": dict}}

    for group_name, queries in query_groups.items():
        weight = group_weights.get(group_name, 1.0)
        for query in queries:
            results = retrieve_documents(
                query=query,
                docs=docs,
                doc_title_embeddings=doc_title_embeddings,
                doc_body_embeddings=doc_body_embeddings,
                embedding_model=embedding_model,
                top_k=top_k_per_query,
            )
            for r in results:
                key = r["title"]
                if key not in score_map:
                    score_map[key] = {"score": 0.0, "hits": 0, "queries": [], "doc": r}
                score_map[key]["score"]  += r["score"] * weight
                score_map[key]["hits"]   += 1
                score_map[key]["queries"].append(f"[{group_name}] {query}")

    sorted_docs = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)

    rows = []
    for item in sorted_docs[:final_top_k]:
        row = item["doc"].copy()
        row["final_score"]      = item["score"]
        row["hit_count"]        = item["hits"]
        row["matched_queries"]  = " | ".join(item["queries"])
        rows.append(row)

    return pd.DataFrame(rows)