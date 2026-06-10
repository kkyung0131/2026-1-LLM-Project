"""
llm.py
LLM 쿼리 생성 및 최종 컨설팅 리포트 생성 모듈

포함 함수:
- get_llm_client()        : OpenAI 호환 클라이언트 초기화
- generate_rag_queries()  : 가맹점 컨텍스트 → 분석 + RAG 검색 쿼리 생성
- build_final_prompt()    : LLM 최종 입력 프롬프트 조합
- generate_rag_report()   : 전체 RAG 파이프라인 실행 → 컨설팅 리포트 반환

시스템 프롬프트는 core/prompts.py 에서 관리합니다.
"""

import os
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from core.context_builder import build_context, build_review_context
from core.prompts import QUERY_SYSTEM_PROMPT, REVIEW_QUERY_SYSTEM_PROMPT, FINAL_SYSTEM_PROMPT
from core.rag import retrieve_with_fusion


# ============================================================
# 클라이언트 초기화
# ============================================================

def get_llm_client(base_url: str = None, api_key: str = None) -> OpenAI:
    """
    OpenAI 호환 클라이언트 초기화
    base_url, api_key 미전달 시 환경변수에서 자동 로드
    """
    return OpenAI(
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key  or os.getenv("OPENAI_API_KEY"),
    )


# ============================================================
# Step 1: RAG 쿼리 생성
# ============================================================

def generate_rag_queries(
    store_name: str,
    df: pd.DataFrame,
    client: OpenAI,
    stream_callback=None,
) -> dict:
    """
    가맹점 컨텍스트 → LLM 분석 + RAG 검색 쿼리 생성

    Args:
        store_name     : 가맹점명
        df             : 전체 DataFrame
        client         : OpenAI 클라이언트
        owner_input    : 사장님 관심 사항 (없으면 None)
        stream_callback: 스트리밍 토큰 콜백 함수

    Returns:
        {
            "store_name": str,
            "context"  : str,
            "response" : str,
            "analysis" : str,
            "queries"  : list[str],
        }
    """
    context = build_context(store_name, df)

    messages = [
        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
        {"role": "user",   "content": f"아래는 가맹점 분석 데이터입니다.\n\n{context}"},
    ]

    stream = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        extra_body={"repetition_penalty": 1.1},
        stream=True,
    )

    response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            response += delta
            if stream_callback:
                stream_callback(delta)

    response = response.strip()
    analysis, queries = "", []

    if "[분석]" in response and "[검색 쿼리]" in response:
        analysis     = response.split("[검색 쿼리]")[0].replace("[분석]", "").strip()
        queries_part = response.split("[검색 쿼리]")[1].strip()
        for line in queries_part.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                query = line.split(".", 1)[-1].strip()
                if query:
                    queries.append(query)

    return {
        "store_name": store_name,
        "context":    context,
        "response":   response,
        "analysis":   analysis,
        "queries":    queries,
    }

def generate_review_rag_queries(
    store_name: str,
    store_df: pd.DataFrame,
    review_threshold: pd.Series,
    client: OpenAI,
    stream_callback=None,
) -> dict:
    """
    리뷰 컨텍스트 → LLM 분석 + RAG 검색 쿼리 생성

    Args:
        store_name       : 가맹점명
        store_df         : 리뷰용 가게별 단일 행 DataFrame
        review_threshold : 업종별 리뷰 수 75% 분위수
        client           : OpenAI 클라이언트
        owner_input      : 사장님 관심 사항 (없으면 None)
        stream_callback  : 스트리밍 토큰 콜백 함수

    Returns:
        {
            "store_name": str,
            "context"  : str,
            "response" : str,
            "analysis" : str,
            "queries"  : list[str],
        }
    """

    context = build_review_context(store_name, store_df, review_threshold)

    messages = [
        {"role": "system", "content": REVIEW_QUERY_SYSTEM_PROMPT},
        {"role": "user",   "content": f"아래는 가맹점 리뷰 데이터입니다.\n\n{context}"},
    ]

    stream = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        extra_body={"repetition_penalty": 1.1},
        stream=True,
    )

    response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            response += delta
            if stream_callback:
                stream_callback(delta)

    response = response.strip()
    analysis, queries = "", []

    if "[분석]" in response and "[검색 쿼리]" in response:
        analysis     = response.split("[검색 쿼리]")[0].replace("[분석]", "").strip()
        queries_part = response.split("[검색 쿼리]")[1].strip()
        for line in queries_part.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                query = line.split(".", 1)[-1].strip()
                if query:
                    queries.append(query)

    return {
        "store_name": store_name,
        "context":    context,
        "response":   response,
        "analysis":   analysis,
        "queries":    queries,
    }


# ============================================================
# Step 2: 최종 프롬프트 조합
# ============================================================

def build_final_prompt(
    store_context: str,
    retrieved_docs: list,
    review_context: str = None,
    owner_input: str = None,
) -> str:
    """
    LLM 최종 입력 프롬프트 조합

    Args:
        store_context : build_context() 결과
        retrieved_docs: retrieve_with_fusion() 결과 records
        review_context: 리뷰 분석 텍스트 (없으면 None)
        owner_input   : 사장님 관심 사항 (없으면 None)
    """
    prompt = "[가맹점 분석 현황]\n" + store_context + "\n\n"

    if review_context:
        prompt += "[고객 리뷰 분석]\n" + review_context + "\n\n"

    if owner_input:
        prompt += "[사장님 관심 사항]\n" + owner_input + "\n\n"

    prompt += "[검색된 마케팅 전략 문서]\n\n"
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += (
            f"[문서 {i}]\n"
            f"제목: {doc['title']}\n"
            f"카테고리: {doc['category_1']} > {doc['category_2']}\n"
            f"내용: {doc['text']}\n\n"
        )

    prompt += "위의 가맹점 분석 데이터, 고객 리뷰 분석, 검색된 마케팅 전략 문서를 기반으로 컨설팅 리포트를 작성하라."
    return prompt


# ============================================================
# Step 3: 전체 RAG 파이프라인
# ============================================================

def generate_rag_report(
    store_name: str,
    df: pd.DataFrame,
    client: OpenAI,
    docs: list,
    doc_title_embeddings: np.ndarray,
    doc_body_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    store_df: pd.DataFrame,          # 추가
    review_threshold: pd.Series,     # 추가
    owner_input: str = None,
    top_k_per_query: int = 3,
    final_top_k: int = 10,
    query_stream_callback=None,
    review_stream_callback=None,
    report_stream_callback=None,
) -> dict:
    """
    전체 RAG 파이프라인 실행

    Args:
        store_name             : 가맹점명
        df                     : 전체 DataFrame
        client                 : OpenAI 클라이언트
        docs                   : 문서 리스트
        doc_title_embeddings   : title 임베딩 행렬
        doc_body_embeddings    : body 임베딩 행렬
        embedding_model        : SentenceTransformer 모델
        owner_input            : 사장님 관심 사항 (없으면 None)
        review_queries         : 리뷰 기반 검색 쿼리 리스트 (없으면 None)
        review_context         : 리뷰 분석 텍스트 (없으면 None)
        top_k_per_query        : 쿼리당 검색 문서 수
        final_top_k            : 최종 사용 문서 수
        query_stream_callback  : 1단계 LLM 스트리밍 콜백 (정형 쿼리 생성)
        review_stream_callback : 2단계 LLM 스트리밍 콜백 (리뷰 쿼리 생성)
        report_stream_callback : 3단계 LLM 스트리밍 콜백 (리포트 생성)

    Returns:
        {
            "store_name"         : str,
            "store_context"      : str,
            "analysis"           : str,
            "query_groups"       : dict,
            "retrieved_documents": list,
            "final_prompt"       : str,
            "report"             : str,
        }
    """
    # Step 1: 가맹점 컨텍스트 + RAG 쿼리 생성
    rag_query_result   = generate_rag_queries(
        store_name, df, client, owner_input,
        stream_callback=query_stream_callback,
    )
    store_context      = rag_query_result["context"]
    structured_queries = rag_query_result["queries"]
    analysis           = rag_query_result["analysis"]

    review_context = None
    review_queries = []

    if store_name in store_df["title"].values:
        review_context = build_review_context(store_name, store_df, review_threshold)
        review_result  = generate_review_rag_queries(
            store_name, store_df, review_threshold, client,
            owner_input=owner_input,
            stream_callback=review_stream_callback,
        )
        review_queries = review_result["queries"]

    # Step 2: 쿼리 그룹 구성 + 문서 검색
    query_groups = {
        "정형": structured_queries,
        "리뷰": review_queries,
    }

    fusion_df     = retrieve_with_fusion(
        query_groups=query_groups,
        docs=docs,
        doc_title_embeddings=doc_title_embeddings,
        doc_body_embeddings=doc_body_embeddings,
        embedding_model=embedding_model,
        top_k_per_query=top_k_per_query,
        final_top_k=final_top_k,
        group_weights={"정형": 1.0, "리뷰": 1.0},
    )
    all_retrieved = fusion_df.to_dict("records")

    # Step 3: 최종 프롬프트 + 리포트 생성
    final_prompt = build_final_prompt(
        store_context=store_context,
        retrieved_docs=all_retrieved,
        review_context=review_context,
        owner_input=owner_input,
    )

    messages = [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user",   "content": final_prompt},
    ]

    stream = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        extra_body={"repetition_penalty": 1.1},
        stream=True,
    )

    report = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            report += delta
            if report_stream_callback:
                report_stream_callback(delta)

    return {
        "store_name":          store_name,
        "store_context":       store_context,
        "review_context":      review_context, 
        "analysis":            analysis,
        "query_groups":        query_groups,
        "retrieved_documents": all_retrieved,
        "final_prompt":        final_prompt,
        "report":              report.strip(),
    }
