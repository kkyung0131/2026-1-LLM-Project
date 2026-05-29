"""
run.py
전체 RAG 파이프라인 CLI 실행 스크립트

사용법:
    python run.py --store "성수AGU"
    python run.py --store "성수AGU" --owner "배달 매출을 늘리고 싶어요"
    python run.py --store "성수AGU" --no-stream          # 스트리밍 없이 결과만 출력
    python run.py --store "성수AGU" --save               # 리포트를 txt 파일로 저장
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from core.llm import get_llm_client, generate_rag_report
from core.rag import load_embedding_model, load_documents

# ============================================================
# 경로 설정
# ============================================================

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"

DF_PATH   = DATA_DIR / "score_review_merged.csv"
JSON_PATH = DATA_DIR / "baemin_articles_embedded.json"


# ============================================================
# 출력 헬퍼
# ============================================================

def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_step(msg: str):
    print(f"\n[→] {msg}")


# ============================================================
# 메인
# ============================================================

def main():
    load_dotenv()

    # ── 인자 파싱 ──
    parser = argparse.ArgumentParser(description="가맹점 컨설팅 리포트 생성")
    parser.add_argument("--store",     required=True,      help="가맹점명 (예: '성수AGU')")
    parser.add_argument("--owner",     default=None,       help="사장님 관심 사항 (선택)")
    parser.add_argument("--top-k",     type=int, default=3,help="쿼리당 검색 문서 수 (기본 3)")
    parser.add_argument("--final-k",   type=int, default=5,help="최종 사용 문서 수 (기본 5)")
    parser.add_argument("--no-stream", action="store_true",help="스트리밍 비활성화")
    parser.add_argument("--save",      action="store_true",help="리포트를 파일로 저장")
    args = parser.parse_args()

    # ── 데이터 로드 ──
    print_step("데이터 로드 중...")
    if not DF_PATH.exists():
        print(f"[오류] 파일 없음: {DF_PATH}")
        sys.exit(1)
    if not JSON_PATH.exists():
        print(f"[오류] 파일 없음: {JSON_PATH}")
        sys.exit(1)

    df                                          = pd.read_csv(DF_PATH)
    docs, doc_title_embeddings, doc_body_embeddings = load_documents(JSON_PATH)

    if args.store not in df["title"].values:
        print(f"[오류] '{args.store}' 가맹점을 찾을 수 없습니다.")
        print("등록된 가맹점 예시:", df["title"].unique()[:5].tolist())
        sys.exit(1)

    # ── 모델 & 클라이언트 초기화 ──
    print_step("임베딩 모델 로드 중...")
    embedding_model = load_embedding_model()

    print_step("LLM 클라이언트 초기화 중...")
    client = get_llm_client()

    # ── 파이프라인 실행 ──
    print_section(f"리포트 생성 시작: {args.store}")
    if args.owner:
        print(f"  사장님 관심 사항: {args.owner}")

    # 스트리밍 콜백 — --no-stream 이면 None
    def stream_print(delta: str):
        print(delta, end="", flush=True)

    query_cb  = None if args.no_stream else stream_print
    report_cb = None if args.no_stream else stream_print

    # 1단계: 쿼리 생성
    print_section("Step 1 | 분석 및 검색 쿼리 생성")
    result = generate_rag_report(
        store_name=args.store,
        df=df,
        client=client,
        docs=docs,
        doc_title_embeddings=doc_title_embeddings,
        doc_body_embeddings=doc_body_embeddings,
        embedding_model=embedding_model,
        owner_input=args.owner,
        top_k_per_query=args.top_k,
        final_top_k=args.final_k,
        query_stream_callback=query_cb,
        report_stream_callback=report_cb,
    )

    # ── 결과 출력 ──
    if args.no_stream:
        print_section("분석")
        print(result["analysis"])

        print_section("생성된 검색 쿼리")
        for i, q in enumerate(result["query_groups"]["정형"], 1):
            print(f"  {i}. {q}")

        print_section("참고 문서")
        for doc in result["retrieved_documents"]:
            print(f"  · [{doc['category_1']} > {doc['category_2']}] {doc['title']}")

        print_section("컨설팅 리포트")
        print(result["report"])

    else:
        # 스트리밍 중에 이미 출력됨 — 참고 문서만 추가 출력
        print_section("참고 문서")
        for doc in result["retrieved_documents"]:
            print(f"  · [{doc['category_1']} > {doc['category_2']}] {doc['title']}")

    # ── 파일 저장 ──
    if args.save:
        save_dir  = ROOT / "output"
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = save_dir / f"{args.store}_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"가맹점: {args.store}\n")
            if args.owner:
                f.write(f"사장님 관심 사항: {args.owner}\n")
            f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            f.write(result["report"])

        print(f"\n[저장 완료] {filename}")


if __name__ == "__main__":
    main()