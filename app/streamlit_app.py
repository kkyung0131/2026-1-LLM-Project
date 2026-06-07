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
    page_title="성동구 소상공인 위기진단 및 컨설팅 AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 커스텀 CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], * {
    font-family: 'Noto Sans KR', sans-serif !important;
}  

/* 사이드바 */
[data-testid="stSidebar"] {
    background: #00462A;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #cccccc !important;
    color: #1a1a1a !important;
    border-radius: 8px !important;
}
            
/* 일반 버튼 */            
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
    font-size: 13px !important;
    padding: 0.3rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    line-height: 1.4 !important;
    min-height: 0 !important;
    height: auto !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255, 255, 255, 0.18) !important;
    border-color: rgba(255, 255, 255, 0.6) !important;
}

/* 리포트 생성 버튼 */
[data-testid="stSidebar"] .run-btn .stButton button {
    background: linear-gradient(135deg, #ff6b35, #f7931e) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    width: 100% !important;
    margin-top: 8px !important;
}

[data-testid="stSidebar"] .run-btn .stButton button p {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: white !important;
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
    border: 1.5px solid #e8ecf0;
    border-radius: 20px;
    padding: 28px 16px 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
}
.gauge-title {
    font-size: 16px;
    font-weight: 600;
    color: #000000;
    margin-bottom: 12px;
    letter-spacing: 0.03em;
}
.gauge-score {
    font-family: 'DM Serif Display', serif;
    font-size: 64px;        /* ← 42px → 64px */
    line-height: 1;
    margin: 8px 0;
}
.gauge-label {
    font-size: 16px;
    font-weight: 700;
    padding: 5px 18px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 8px;
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
    font-family: 'DM Serif Display', serif !important;
    font-size: 48px;
    line-height: 1.2;
    color: #00462A;
    margin-bottom: 12px;
    font-weight: 700 !important;  
}
.hero-sub {
    font-size: 20px;
    color: #000000;
    line-height: 1.7;
}
.step-card {
    background: #f8fafc;
    border-left: 3px solid #00462A;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.step-num {
    font-size: 14px;
    font-weight: 700;
    color: #00462A;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.step-desc {
    font-size: 17px;
    color: #000000;
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
        <div style="font-size:15px; font-weight:600; color:#000000; margin-bottom:6px;">/ 100점</div>
        <span class="gauge-label {lc}">{level}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:
    st.markdown("### 🍽️ 성동구 소상공인 AI 컨설턴트")
    st.markdown('<hr style="margin: 6px 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">', unsafe_allow_html=True)
    
    st.markdown("**🏪 가맹점명**")
    store_name = st.text_input(
        "",
        placeholder="예) 악어떡볶이, 빽다방 상왕십리역점",
        help="데이터에 등록된 가맹점명을 정확히 입력하세요.",
        label_visibility="collapsed",  
    )
    st.markdown('<hr style="margin: 6px 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">', unsafe_allow_html=True)

    st.markdown("❓ **궁금한 점**")
    if "owner_input" not in st.session_state:
        st.session_state["owner_input"] = ""
    chips = ["신메뉴 개발 방법 알려줘", 
             "SNS 인기 매장이 되고 싶어", 
             "배달 매출을 키우고 싶어",
             "리뷰 점수를 올리고 싶어"]
    for i, chip in enumerate(chips):
        if st.button(chip, key=f"chip_{i}", use_container_width=True):
            st.session_state["owner_input"] = chip

    owner_input = st.text_area(
        "",
        value=st.session_state["owner_input"],
        placeholder="사장님의 고민이나 목표를 자유롭게 입력하세요. (생략 가능)",
        height=100,
        label_visibility="collapsed",
    )
    st.markdown('<hr style="margin: 6px 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">', unsafe_allow_html=True)

    with st.expander("⚙️ 고급 설정", expanded=False):
        top_k   = st.slider("쿼리당 검색 문서 수", 1, 5, 3)   # min, max, default
        final_k = st.slider("최종 참고 문서 수",   3, 20, 10)  # min, max, default
    
    st.markdown('<hr style="margin: 6px 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">', unsafe_allow_html=True)
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run_btn = st.button("📊 리포트 생성", key="run_btn", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 메인 화면
# ============================================================

# ── 초기 화면 (가맹점명 미입력 상태) ──
if not store_name and not run_btn:
    st.markdown('<div class="hero-title">성동구 소상공인을 위한<br> 위기진단 및 마케팅 전략 AI 컨설턴트</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">매출·고객·상권·평판 데이터를 분석해 맞춤형 전략 리포트를 생성합니다.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 사용 방법")

    steps = [
        ("STEP 01", "왼쪽 사이드바에 <strong>가맹점명</strong>을 입력합니다."),                              # ← ** → <strong>
        ("STEP 02", "궁금한 점이나 고민이 있다면 <strong>궁금한 점</strong>에 자유롭게 적어주세요."),         # ← ** → <strong>
        ("STEP 03", "<strong>리포트 생성</strong> 버튼을 누르면 AI가 분석을 시작합니다."),                   # ← ** → <strong>
        ("STEP 04", "위험 스코어 대시보드와 <strong>맞춤 컨설팅 리포트</strong>를 확인하세요."),             # ← ** → <strong>
    ]
    for num, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
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
        use_cols = [
            "id", "ym_quarter", "big_ind", "dong", "title", "category",
            "score", "score_taste", "score_price", "score_service",
            "review_cnt", "reviews",
        ]
        store_df         = df[use_cols].drop_duplicates(subset=["id"]).copy()
        review_threshold = store_df.groupby("big_ind")["review_cnt"].quantile(0.75)
    
    if store_name not in df["title"].values:
        st.error(f"'{store_name}' 가맹점을 찾을 수 없습니다. 가맹점명을 다시 확인해주세요.")
        st.stop()

    # ── 위험 스코어 먼저 표시 ──
    latest_q  = df[df["title"] == store_name]["ym_quarter"].max()
    row       = df[(df["title"] == store_name) & (df["ym_quarter"] == latest_q)].iloc[0]
    risk_info = get_risk_text(row, df)

    st.markdown(f"## 🔎 <span style='color:#00462A; font-weight:700;'>{store_name} 위험 진단", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:16px; color:#000000; margin-top:-8px;'>기준 분기: {latest_q} · 업종: {row['big_ind']} · 위치: {row['dong']}</p>", unsafe_allow_html=True)
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
    <div style="background:#f8fafc; border-radius:12px; padding:16px 20px;
                border-left:4px solid #00462A;">
        <span style="font-size:18px; font-weight:700; color:#000000;">종합 진단 &nbsp;|&nbsp;</span>
        <span style="font-size:18px; font-weight:700; color:#000000;">{diagnosis}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 리포트 스트리밍 ──
    st.markdown("## 💡 <span style='color:#00462A; font-weight:700;'>AI 컨설팅 리포트</span>", unsafe_allow_html=True)
    if owner_input:
        st.caption(f"사장님 관심 사항 반영: *{owner_input}*")

    report_placeholder = st.empty()
    report_text = ""

    def report_cb(delta):
        global report_text
        report_text += delta
        report_placeholder.markdown(
            f'<div class="report-container">{report_text}</div>',
            unsafe_allow_html=True,
        )

    with st.spinner("AI가 분석 중입니다..."):
        result = generate_rag_report(
            store_name=store_name,
            df=df,
            client=client,
            docs=docs,
            doc_title_embeddings=doc_title_embeddings,
            doc_body_embeddings=doc_body_embeddings,
            embedding_model=embedding_model,
            store_df=store_df,                   
            review_threshold=review_threshold,   
            owner_input=owner_input or None,
            top_k_per_query=top_k,
            final_top_k=final_k,
            query_stream_callback=None,
            report_stream_callback=report_cb,
        )

    # 최종 리포트 (스트리밍 끝난 후 마크다운 렌더링)
    report_placeholder.markdown(result["report"])

    # ── 참고 문서 출처 ──
    st.markdown("---")
    st.markdown("## <span style='color:#00462A;'>📚 참고 문서</span>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:16px; color:#000000; margin-top:-8px;'>본 리포트는 아래 문서를 기반으로 작성되었습니다.</p>", unsafe_allow_html=True)

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
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**📊 정형 데이터 기반 쿼리**")
            for i, q in enumerate(result["query_groups"]["정형"], 1):
                st.markdown(f"`{i}.` {q}")

        with col_right:
            if result["query_groups"].get("리뷰"):
                st.markdown("**📝 리뷰 데이터 기반 쿼리**")
                for i, q in enumerate(result["query_groups"]["리뷰"], 1):
                    st.markdown(f"`{i}.` {q}")