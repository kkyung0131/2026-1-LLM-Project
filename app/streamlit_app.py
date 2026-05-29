"""
streamlit_app.py
가맹점 컨설팅 리포트 대시보드

실행: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.context_builder import build_context, get_risk_text
from core.llm import get_llm_client, generate_rag_report
from core.rag import load_embedding_model, load_documents

load_dotenv()

# ============================================================
# 경로
# ============================================================
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
DF_PATH   = DATA_DIR / "score_review_merged.csv"
JSON_PATH = DATA_DIR / "baemin_articles_embedded.json"

# ============================================================
# 캐시 — 무거운 리소스는 한 번만 로드
# ============================================================

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()

@st.cache_resource
def get_docs():
    return load_documents(JSON_PATH)

@st.cache_resource
def get_client():
    return get_llm_client()

@st.cache_data
def get_df():
    return pd.read_csv(DF_PATH)

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="가맹점 컨설팅 AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 커스텀 CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: #1a1d27 !important;
    border: 1px solid #2e3245 !important;
    color: #fff !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #ff6b35, #f7931e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    opacity: 0.85 !important;
}

/* 예시 칩 버튼 */
.example-chip {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2e3245;
    color: #a0aec0 !important;
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 12px;
    margin: 3px 2px;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
}
.example-chip:hover {
    background: #2e3245;
    color: #fff !important;
}

/* 위험 스코어 게이지 카드 */
.gauge-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 16px;
    padding: 20px 16px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.gauge-title {
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
    margin-bottom: 8px;
    letter-spacing: 0.03em;
}
.gauge-score {
    font-family: 'DM Serif Display', serif;
    font-size: 42px;
    line-height: 1;
    margin: 6px 0;
}
.gauge-label {
    font-size: 13px;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 4px;
}
.level-safe   { color: #16a34a; background: #dcfce7; }
.level-warn   { color: #d97706; background: #fef3c7; }
.level-danger { color: #dc2626; background: #fee2e2; }
.score-safe   { color: #16a34a; }
.score-warn   { color: #d97706; }
.score-danger { color: #dc2626; }

/* 리포트 컨테이너 */
.report-container {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 16px;
    padding: 32px 36px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    line-height: 1.85;
}

/* 참고 문서 태그 */
.doc-tag {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    color: #475569;
    margin: 4px;
}
.doc-tag-cat {
    font-size: 11px;
    color: #94a3b8;
    display: block;
    margin-bottom: 2px;
}

/* 히어로 섹션 */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 48px;
    line-height: 1.15;
    color: #0f1117;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 17px;
    color: #64748b;
    line-height: 1.7;
}
.step-card {
    background: #f8fafc;
    border-left: 3px solid #ff6b35;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.step-num {
    font-size: 11px;
    font-weight: 700;
    color: #ff6b35;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.step-desc {
    font-size: 14px;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 헬퍼: 위험 수준 → CSS 클래스
# ============================================================

def level_class(level: str) -> tuple[str, str]:
    """(label_class, score_class) 반환"""
    return {
        "안정": ("level-safe",   "score-safe"),
        "주의": ("level-warn",   "score-warn"),
        "위험": ("level-danger", "score-danger"),
    }.get(level, ("level-safe", "score-safe"))


def render_gauge(title: str, score: float, level: str):
    lc, sc = level_class(level)
    st.markdown(f"""
    <div class="gauge-card">
        <div class="gauge-title">{title}</div>
        <div class="gauge-score {sc}">{score:.0f}</div>
        <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">/ 100점</div>
        <span class="gauge-label {lc}">{level}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:
    st.markdown("### 🍽️ 가맹점 컨설팅 AI")
    st.markdown("---")

    store_name = st.text_input(
        "가맹점명",
        placeholder="예) 성수AGU, 악어떡볶이",
        help="데이터에 등록된 가맹점명을 정확히 입력하세요.",
    )

    st.markdown("**💬 궁금한 점 (선택)**")
    st.markdown(
        '<span class="example-chip">신메뉴 개발 방법 알려줘</span>'
        '<span class="example-chip">SNS 인기 매장이 되고 싶어</span>'
        '<span class="example-chip">단골 고객을 늘리고 싶어</span>'
        '<span class="example-chip">배달 매출을 키우고 싶어</span>',
        unsafe_allow_html=True,
    )
    owner_input = st.text_area(
        "",
        placeholder="사장님의 고민이나 목표를 자유롭게 입력하세요. (생략 가능)",
        height=100,
        label_visibility="collapsed",
    )

    st.markdown("---")
    run_btn = st.button("📊 리포트 생성", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ 고급 설정"):
        top_k   = st.slider("쿼리당 검색 문서 수", 1, 5, 3)
        final_k = st.slider("최종 참고 문서 수",   3, 10, 5)


# ============================================================
# 메인 화면
# ============================================================

# ── 초기 화면 (가맹점명 미입력 상태) ──
if not store_name and not run_btn:
    st.markdown('<div class="hero-title">외식업 가맹점을<br>위한 AI 컨설턴트</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">매출·고객·상권·평판 데이터를 분석해<br>맞춤형 전략 리포트를 즉시 생성합니다.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 이용 방법")

    steps = [
        ("STEP 01", "왼쪽 사이드바에 **가맹점명**을 입력합니다."),
        ("STEP 02", "궁금한 점이나 고민이 있다면 **관심 사항**에 자유롭게 적어주세요."),
        ("STEP 03", "**리포트 생성** 버튼을 누르면 AI가 분석을 시작합니다."),
        ("STEP 04", "위험 스코어 대시보드와 **맞춤 컨설팅 리포트**를 확인하세요."),
    ]
    for num, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **궁금한 점 예시**: '신메뉴 개발 방법 알려줘' / 'SNS 인기 매장이 되고 싶어' / '배달 매출을 키우고 싶어'")
    st.stop()


# ── 가맹점명 입력됐지만 버튼 안 누른 상태 ──
if store_name and not run_btn:
    st.markdown(f"### `{store_name}` 분석 준비 완료")
    st.markdown("사이드바의 **리포트 생성** 버튼을 눌러주세요.")
    st.stop()


# ── 리포트 생성 실행 ──
if run_btn:
    if not store_name:
        st.error("가맹점명을 입력해주세요.")
        st.stop()

    # 데이터 로드
    with st.spinner("데이터 로드 중..."):
        df                                              = get_df()
        docs, doc_title_embeddings, doc_body_embeddings = get_docs()
        embedding_model                                 = get_embedding_model()
        client                                          = get_client()

    if store_name not in df["title"].values:
        st.error(f"'{store_name}' 가맹점을 찾을 수 없습니다. 가맹점명을 다시 확인해주세요.")
        st.stop()

    # ── 위험 스코어 먼저 표시 ──
    latest_q  = df[df["title"] == store_name]["ym_quarter"].max()
    row       = df[(df["title"] == store_name) & (df["ym_quarter"] == latest_q)].iloc[0]
    risk_info = get_risk_text(row, df)

    st.markdown(f"## 📊 {store_name} 위험 진단")
    st.caption(f"기준 분기: {latest_q}  ·  업종: {row['big_ind']}  ·  위치: {row['dong']}")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_gauge("종합 위험도", row["final_risk_score"], risk_info["final_level"])
    with col2:
        render_gauge("매출 위험도", row["sales_score"], risk_info["sales_level"])
    with col3:
        render_gauge("고객 위험도", row["cust_score"], risk_info["cust_level"])
    with col4:
        render_gauge("상권 위험도", row["mkt_score"], risk_info["mkt_level"])

    # 종합 진단 한 줄
    st.markdown("<br>", unsafe_allow_html=True)
    diagnosis = risk_info["combo"] or "전반적으로 안정적인 상태입니다."
    final_lc, _ = level_class(risk_info["final_level"])
    st.markdown(f"""
    <div style="background:#f8fafc;border-radius:12px;padding:14px 20px;
                border-left:4px solid {'#dc2626' if risk_info['final_level']=='위험' else '#d97706' if risk_info['final_level']=='주의' else '#16a34a'}">
        <span style="font-size:13px;font-weight:700;color:#64748b;">종합 진단 &nbsp;|&nbsp;</span>
        <span style="font-size:14px;color:#1e293b;">{diagnosis}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 리포트 스트리밍 ──
    st.markdown("## 💡 AI 컨설팅 리포트")
    if owner_input:
        st.caption(f"사장님 관심 사항 반영: *{owner_input}*")

    report_placeholder = st.empty()
    report_text = ""

    # 쿼리 생성 단계 상태 표시
    with st.status("AI 분석 중...", expanded=False) as status:
        query_log = st.empty()
        query_buf = ""

        def query_cb(delta):
            global query_buf
            query_buf += delta
            query_log.markdown(f"```\n{query_buf[-800:]}\n```")

        def report_cb(delta):
            global report_text
            report_text += delta
            report_placeholder.markdown(
                f'<div class="report-container">{report_text}</div>',
                unsafe_allow_html=True,
            )

        result = generate_rag_report(
            store_name=store_name,
            df=df,
            client=client,
            docs=docs,
            doc_title_embeddings=doc_title_embeddings,
            doc_body_embeddings=doc_body_embeddings,
            embedding_model=embedding_model,
            owner_input=owner_input or None,
            top_k_per_query=top_k,
            final_top_k=final_k,
            query_stream_callback=query_cb,
            report_stream_callback=report_cb,
        )
        status.update(label="분석 완료 ✅", state="complete", expanded=False)

    # 최종 리포트 (스트리밍 끝난 후 마크다운 렌더링)
    report_placeholder.markdown(result["report"])

    # ── 참고 문서 출처 ──
    st.markdown("---")
    st.markdown("#### 📚 참고 문서")
    st.caption("본 리포트는 아래 문서를 기반으로 작성되었습니다.")

    doc_cols = st.columns(3)
    for i, doc in enumerate(result["retrieved_documents"]):
        with doc_cols[i % 3]:
            st.markdown(f"""
            <div class="doc-tag">
                <span class="doc-tag-cat">{doc['category_1']} &rsaquo; {doc['category_2']}</span>
                {doc['title']}
            </div>
            """, unsafe_allow_html=True)

    # ── 생성된 검색 쿼리 (접어두기) ──
    with st.expander("🔍 AI가 생성한 검색 쿼리 보기"):
        for i, q in enumerate(result["query_groups"]["정형"], 1):
            st.markdown(f"`{i}.` {q}")