"""
context_builder.py
가맹점 분석 컨텍스트 생성 모듈

포함 함수:
- sales_to_text()       : 매출 성과 텍스트
- customer_to_text()    : 고객 상황 텍스트
- district_to_text()    : 상권 상황 텍스트
- get_reputation_text() : 고객 평판 텍스트
- get_risk_text()       : 위험 진단 텍스트
- build_context()       : 위 5개를 조합한 최종 컨텍스트 문자열 반환
"""

import numpy as np
import pandas as pd


# ============================================================
# 공통 유틸
# ============================================================

def rank_to_tier(rank: int, total: int) -> str:
    pct = rank / total
    if pct <= 0.33:
        return "상위권"
    elif pct <= 0.66:
        return "중위권"
    else:
        return "하위권"


def get_dong_rank(row, full_df: pd.DataFrame, col: str) -> dict:
    """dong 기준 특정 컬럼의 순위 반환 (높은 순 / 낮은 순)"""
    latest_quarter = full_df["ym_quarter"].max()
    latest_df = full_df[full_df["ym_quarter"] == latest_quarter]

    dong_df = latest_df.drop_duplicates(subset="dong")[["dong", col]]
    total = len(dong_df)

    high_rank = dong_df.sort_values(col, ascending=False).reset_index(drop=True)
    rank_high = high_rank[high_rank["dong"] == row["dong"]].index[0] + 1
    rank_low  = total - rank_high + 1

    return {"rank_high": rank_high, "rank_low": rank_low, "total": total}


# ============================================================
# 1. 매출 성과
# ============================================================

def get_current_level(val: float) -> dict:
    """
    카테고리 평균 순위값(낮을수록 좋음) 기준 현재 수준 반환
    Returns:
        label  : "매우 높음" / "높음" / "보통" / "낮음" / "매우 낮음"
        summary: "높음" / "보통" / "낮음"
    """
    if val < 2.0:
        return {"label": "매우 높음", "summary": "높음"}
    elif val < 3.0:
        return {"label": "높음",     "summary": "높음"}
    elif val < 4.0:
        return {"label": "보통",     "summary": "보통"}
    elif val < 5.0:
        return {"label": "낮음",     "summary": "낮음"}
    else:
        return {"label": "매우 낮음","summary": "낮음"}


def get_long_term_trend(quarters: list) -> dict:
    """
    장기 추이 분석 (첫 분기 vs 마지막 분기)
    값이 낮아질수록 성과 좋아짐 → diff 음수 = 상승세
    Returns:
        text   : 장기 추이 텍스트
        summary: "상승" / "보합" / "하락"
    """
    diff = quarters[-1] - quarters[0]

    if diff >= 1.5:
        return {"text": "장기 뚜렷한 하락세",  "summary": "하락"}
    elif diff >= 0.5:
        return {"text": "장기 완만한 하락세",  "summary": "하락"}
    elif diff >= -0.5:
        return {"text": "장기 보합세",         "summary": "보합"}
    elif diff >= -1.5:
        return {"text": "장기 완만한 상승세",  "summary": "상승"}
    else:
        return {"text": "장기 뚜렷한 상승세",  "summary": "상승"}


def get_level_combination(amt_summary: str, cnt_summary: str, aov_summary: str) -> str | None:
    high, low = "높음", "낮음"
    if amt_summary == low  and cnt_summary == low  and aov_summary == low:
        return "전반적으로 매출이 부진한 구조입니다"
    elif amt_summary == low  and cnt_summary == low  and aov_summary != low:
        return "방문 고객 수 자체가 부족한 구조입니다"
    elif amt_summary == low  and cnt_summary != low  and aov_summary == low:
        return "방문은 있으나 객단가가 매출을 끌어내리는 구조입니다"
    elif amt_summary != low  and cnt_summary == low  and aov_summary == high:
        return "소수 고객의 고액 결제에 의존하는 구조입니다"
    elif amt_summary == high and cnt_summary == high and aov_summary == high:
        return "전반적으로 매출이 양호한 구조입니다"
    return None


def get_trend_combination_msg(amt_trend: str, cnt_trend: str, aov_trend: str) -> str | None:
    up, down, flat = "상승", "하락", "보합"
    if amt_trend == down and cnt_trend == down and aov_trend == down:
        return "전방위적으로 매출이 악화된 추세입니다."
    elif amt_trend == down and cnt_trend == down and aov_trend != down:
        return "매출 하락이 방문 감소에서 기인했을 가능성이 있습니다."
    elif amt_trend == down and cnt_trend != down and aov_trend == down:
        return "매출 하락이 객단가 하락에서 기인했을 가능성이 있습니다."
    elif amt_trend == flat and cnt_trend == up   and aov_trend == down:
        return "건수 늘었으나 객단가 희석으로 매출이 정체된 상황입니다."
    elif amt_trend == up   and cnt_trend == up   and aov_trend == up:
        return "전반적으로 매출이 성장세를 보입니다."
    return None


def get_sales_percentile_text(df: pd.DataFrame, store_id: str, min_peer: int = 5) -> str:
    """동종업종+상권 내 매출 변수 백분위 텍스트 반환 (peer 부족 시 업종만으로 fallback)"""
    store_row = df[df["id"] == store_id].sort_values("ym_quarter").iloc[-1]
    quarter   = store_row["ym_quarter"]
    big_ind   = store_row["big_ind"]
    dong      = store_row["dong"]

    var_labels = {
        "sales_amt_cat_mean": "매출액",
        "sales_cnt_cat_mean": "매출건수",
        "aov_cat_mean":       "객단가",
    }

    def calc_percentile(peer_df, var, store_val):
        peer_vals = peer_df[var].dropna()
        pct = (peer_vals > store_val).sum() / len(peer_vals) * 100
        return round(pct, 1), len(peer_vals)

    def pct_to_text(pct, label, scope_text):
        if pct >= 80:   level = "높은 수준"
        elif pct >= 60: level = "다소 높은 수준"
        elif pct >= 40: level = "평균 수준"
        elif pct >= 20: level = "다소 낮은 수준"
        else:           level = "낮은 수준"
        return f"{scope_text} 내에서 {label}은 {level}"

    peer_local = df[(df["big_ind"] == big_ind) & (df["dong"] == dong) & (df["ym_quarter"] == quarter)]
    peer_ind   = df[(df["big_ind"] == big_ind) & (df["ym_quarter"] == quarter)]

    texts, peer_cnt = [], 0
    for var, label in var_labels.items():
        store_val = store_row[var]
        if len(peer_local) >= min_peer:
            pct, peer_cnt = calc_percentile(peer_local, var, store_val)
            scope = "동일 업종/상권"
        else:
            pct, peer_cnt = calc_percentile(peer_ind, var, store_val)
            scope = "동일 업종"
        texts.append(pct_to_text(pct, label, scope))

    return ", ".join(texts) + f"입니다. (비교 가맹점 수 {peer_cnt}개 기준)"


def sales_to_text(
    sales_amt_quarters: list,
    sales_cnt_quarters: list,
    aov_quarters: list,
    store_id: str,
    full_df: pd.DataFrame,
) -> dict:
    amt_level = get_current_level(sales_amt_quarters[-1])
    cnt_level = get_current_level(sales_cnt_quarters[-1])
    aov_level = get_current_level(aov_quarters[-1])

    amt_trend = get_long_term_trend(sales_amt_quarters)
    cnt_trend = get_long_term_trend(sales_cnt_quarters)
    aov_trend = get_long_term_trend(aov_quarters)

    level_combo = get_level_combination(amt_level["summary"], cnt_level["summary"], aov_level["summary"])
    trend_combo = get_trend_combination_msg(amt_trend["summary"], cnt_trend["summary"], aov_trend["summary"])
    peer_text   = get_sales_percentile_text(full_df, store_id)

    current_text = (
        f"매출액은 {amt_level['label']}인 수준이며, "
        f"매출건수는 {cnt_level['label']}인 수준, "
        f"객단가는 {aov_level['label']}인 수준입니다."
    )
    if level_combo:
        current_text += f" {level_combo}."

    trend_text = (
        f"장기적으로 매출액은 {amt_trend['text'].replace('장기 ', '')}이고, "
        f"매출건수는 {cnt_trend['text'].replace('장기 ', '')}, "
        f"객단가는 {aov_trend['text'].replace('장기 ', '')}입니다."
    )
    if trend_combo:
        trend_text += f" {trend_combo}."

    full_text = "\n".join([
        "[매출 성과]", "",
        "1. 현재",   current_text, "",
        "2. 추이",   trend_text, "",
        "3. 동종업계 비교", peer_text,
    ])

    return {
        "current_text": current_text,
        "trend_text":   trend_text,
        "level_combo":  level_combo,
        "trend_combo":  trend_combo,
        "peer_text":    peer_text,
        "full_text":    full_text,
    }


# ============================================================
# 2. 고객 상황
# ============================================================

def get_loyalty_combo(re_high: bool, new_high) -> str:
    combo_map = {
        (False, True):  "즉, 신규 고객 유입은 활발하나 재방문으로 이어지지 않는 구조입니다",
        (False, None):  "즉, 재방문율이 낮고 신규 유입도 평범한 구조입니다",
        (False, False): "즉, 신규 유입과 재방문 모두 저조한 구조입니다",
        (True,  True):  "즉, 신규 유입과 재방문 모두 활발한 구조입니다",
        (True,  None):  "즉, 단골 중심 점포, 신규 유입은 평균 수준인 구조입니다",
        (True,  False): "즉, 단골 중심 점포, 신규 고객 유입은 부족한 구조입니다",
    }
    return combo_map[(re_high, new_high)]


def get_loyalty_text(re_cust: float, new_cust: float, full_df: pd.DataFrame) -> dict:
    latest_df = full_df[full_df["ym_quarter"] == full_df["ym_quarter"].max()]

    re_q2  = latest_df["re_cust_rat_mean"].quantile(0.5)
    re_high = re_cust >= re_q2
    re_label = "높은 편" if re_high else "낮은 편"

    new_q1 = latest_df["new_cust_rat_mean"].quantile(0.25)
    new_q3 = latest_df["new_cust_rat_mean"].quantile(0.75)

    if new_cust < new_q1:
        new_label, new_high = "낮음", False
    elif new_cust < new_q3:
        new_label, new_high = "평균 수준", None
    else:
        new_label, new_high = "높음", True

    combo = get_loyalty_combo(re_high, new_high)
    current_text = (
        f"재방문 고객 비율은 동종업계 대비 {re_label}이며, "
        f"신규 고객 유입은 {new_label}입니다. {combo}."
    )

    return {"current_text": current_text, "combo": combo, "re_high": re_high, "new_high": new_high}


def get_gender_text(f_rat: float) -> str | None:
    if f_rat >= 65:   return "여성 고객이 주를 이루는 점포입니다"
    elif f_rat >= 55: return "여성 고객 비중이 다소 높은 점포입니다"
    elif f_rat >= 45: return None
    elif f_rat >= 35: return "남성 고객 비중이 다소 높은 점포입니다"
    else:             return "남성 고객이 주를 이루는 점포입니다"


def get_age_text(row) -> str:
    age_cols = {
        "20대 이하": row["age_20_under_rat_mean"],
        "30대":      row["age_30_rat_mean"],
        "40대":      row["age_40_rat_mean"],
        "50대":      row["age_50_rat_mean"],
        "60대 이상": row["age_60_over_rat_mean"],
    }
    sorted_ages = sorted(age_cols.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_val = sorted_ages[0]
    top2_name, _        = sorted_ages[1]

    if top1_val >= 50:
        return f"{top1_name}가 절반 이상을 차지하는 연령 집중 점포"
    return f"{top1_name}와 {top2_name}가 주력 고객층"


def get_age_label(row) -> str:
    """요약 한 줄용 짧은 레이블"""
    age_cols = {
        "20대 이하": row["age_20_under_rat_mean"],
        "30대":      row["age_30_rat_mean"],
        "40대":      row["age_40_rat_mean"],
        "50대":      row["age_50_rat_mean"],
        "60대 이상": row["age_60_over_rat_mean"],
    }
    sorted_ages = sorted(age_cols.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_val = sorted_ages[0]
    top2_name, _        = sorted_ages[1]

    return top1_name if top1_val >= 50 else f"{top1_name}·{top2_name}"


def get_customer_segment_text(row) -> dict:
    gender_text  = get_gender_text(row["f_rat_mean"])
    age_text     = get_age_text(row)
    current_text = f"{age_text}이며, {gender_text}." if gender_text else f"{age_text}."

    return {"current_text": current_text, "gender_note": gender_text, "age_text": age_text}


def get_customer_type_text(row) -> dict:
    type_cols = {
        "거주": row["resid_cust_rat_mean"],
        "직장": row["office_cust_rat_mean"],
        "유동": row["move_cust_rat_mean"],
    }
    top_type, top_val = max(type_cols.items(), key=lambda x: x[1])

    type_desc_map = {
        "거주": "동네 단골 기반의 점포",
        "직장": "직장인 점심·저녁 수요에 의존하는 점포",
        "유동": "유동 인구 기반으로 1회성 방문 비중이 높은 점포",
    }

    if top_val >= 50:
        label = f"{type_desc_map[top_type]}로, {top_type} 고객이 절반 이상을 차지합니다."
    elif top_val >= 40:
        label = f"{type_desc_map[top_type]}로, {top_type} 고객 비중이 다소 높습니다."
    else:
        label = "거주·직장·유동 고객이 고르게 분포된 점포입니다."

    return {"current_text": label, "top_type": top_type, "top_val": top_val, "label": label}


def get_customer_summary(loyalty: dict, segment: dict, cust_type: dict, row) -> str:
    parts = []
    re_high, new_high = loyalty["re_high"], loyalty["new_high"]

    if re_high and new_high is True:
        parts.append("단골 + 신규 균형")
    elif re_high:
        parts.append("단골 고객")
    elif new_high is True:
        parts.append("신규 유입 활발")

    parts.append(get_age_label(row))

    gender = segment["gender_note"]
    if gender and "여성" in gender:
        parts.append("여성")
    elif gender and "남성" in gender:
        parts.append("남성")

    parts.append(cust_type["top_type"])
    return "주요 고객층: " + " · ".join(parts) + " 고객"


def customer_to_text(re_cust: float, new_cust: float, row, full_df: pd.DataFrame) -> dict:
    loyalty   = get_loyalty_text(re_cust, new_cust, full_df)
    segment   = get_customer_segment_text(row)
    cust_type = get_customer_type_text(row)
    summary   = get_customer_summary(loyalty, segment, cust_type, row)

    full_text = "\n".join([
        "[고객 상황]",
        summary, "",
        "1. 고객 충성도", loyalty["current_text"], "",
        "2. 주요 고객층", segment["current_text"], "",
        "3. 고객 유형",   cust_type["current_text"],
    ])

    return {
        "loyalty_text":   loyalty["current_text"],
        "loyalty_combo":  loyalty["combo"],
        "re_high":        loyalty["re_high"],
        "new_high":       loyalty["new_high"],
        "segment_text":   segment["current_text"],
        "cust_type_text": cust_type["current_text"],
        "summary":        summary,
        "full_text":      full_text,
    }


# ============================================================
# 3. 상권 상황
# ============================================================

def get_district_scale_text(row, full_df: pd.DataFrame) -> dict:
    scale_cols = {
        "총 인구":   "pop_all_mean",
        "직장 인구": "office_num",
        "유동 인구": "move_num",
        "거주 인구": "resid_num",
    }
    lines = []
    for label, col in scale_cols.items():
        r    = get_dong_rank(row, full_df, col)
        tier = rank_to_tier(r["rank_high"], r["total"])
        lines.append(
            f"{label}는 전체 {r['total']}개 동 중 {r['rank_high']}위로 {tier}에 속하는 상권입니다."
        )
    return {"current_text": "\n".join(lines)}


def get_competition_combo(sim_high: bool, franchise_label: str) -> str:
    if sim_high and franchise_label == "높음":
        return "직접 경쟁 및 브랜드 경쟁 모두 치열한 상권"
    elif sim_high and franchise_label in ("낮음", "프랜차이즈 없음"):
        return "개인 업체 간 경쟁이 치열한 상권"
    elif not sim_high and franchise_label == "높음":
        return "프랜차이즈 브랜드 중심 경쟁 상권"
    else:
        return "경쟁 강도가 낮은 상권"


def get_competition_text(row, full_df: pd.DataFrame) -> dict:
    latest_df     = full_df[full_df["ym_quarter"] == full_df["ym_quarter"].max()]
    store_cnt     = row["store_cnt"]
    sim_store_cnt = row["sim_store_cnt"]
    franchise_cnt = row["franchise_cnt"]

    sim_high = sim_store_cnt >= latest_df["sim_store_cnt"].quantile(0.5)
    fran_q2  = latest_df["franchise_cnt"].quantile(0.5)

    if franchise_cnt == 0:
        franchise_label = "프랜차이즈 없음"
    elif franchise_cnt >= fran_q2:
        franchise_label = "높음"
    else:
        franchise_label = "낮음"

    combo    = get_competition_combo(sim_high, franchise_label)
    sim_text = "동종업계 대비 많은 편" if sim_high else "동종업계 대비 적은 편"
    fran_text_map = {
        "높음":           "프랜차이즈 점포가 많아 브랜드 경쟁이 치열합니다",
        "낮음":           "프랜차이즈 점포는 적은 편입니다",
        "프랜차이즈 없음": "프랜차이즈 점포는 없습니다",
    }
    current_text = (
        f"{combo}입니다. 유사업종 수는 {sim_text}이며, {fran_text_map[franchise_label]}."
    )

    return {
        "current_text":    current_text,
        "sim_high":        sim_high,
        "franchise_label": franchise_label,
        "combo":           combo,
    }


def get_rent_text(row, full_df: pd.DataFrame, competition_result: dict) -> dict:
    latest_df = full_df[full_df["ym_quarter"] == full_df["ym_quarter"].max()]
    rent      = row["rent"]
    rent_q1   = latest_df["rent"].quantile(0.25)
    rent_q3   = latest_df["rent"].quantile(0.75)

    if rent >= rent_q3:
        rent_label, rent_desc = "높음", "임대료 부담이 높은"
    elif rent >= rent_q1:
        rent_label, rent_desc = "보통", "임대료 부담이 보통 수준인"
    else:
        rent_label, rent_desc = "낮음", "임대료 부담이 낮은"

    r = get_dong_rank(row, full_df, "rent")

    if rent_label == "낮음":
        tier = rank_to_tier(r["rank_low"], r["total"])
        rank_used = r["rank_low"]
    else:
        tier = rank_to_tier(r["rank_high"], r["total"])
        rank_used = r["rank_high"]

    rent_text = (
        f"임대료는 전체 {r['total']}개 동 중 {rank_used}위로 "
        f"{tier}에 속하며, {rent_desc} 상권입니다."
    )

    is_high_competition = competition_result["combo"] in [
        "직접 경쟁 및 브랜드 경쟁 모두 치열한 상권",
        "개인 업체 간 경쟁이 치열한 상권",
        "프랜차이즈 브랜드 중심 경쟁 상권",
    ]
    combo = None
    if rent_label == "높음" and is_high_competition:
        combo = "경쟁이 치열한 상권에서 임대료까지 높아 고정비 부담이 큰 고위험 상권입니다."
        rent_text += f" {combo}"

    return {"current_text": rent_text, "rent_label": rent_label, "combo": combo}


def district_to_text(row, full_df: pd.DataFrame) -> dict:
    scale       = get_district_scale_text(row, full_df)
    competition = get_competition_text(row, full_df)
    rent        = get_rent_text(row, full_df, competition)

    full_text = "\n".join([
        "[상권 상황]", "",
        "1. 상권 규모",  scale["current_text"], "",
        "2. 경쟁 강도",  competition["current_text"], "",
        "3. 임대 부담",  rent["current_text"],
    ])

    return {
        "scale_text":       scale["current_text"],
        "competition_text": competition["current_text"],
        "rent_text":        rent["current_text"],
        "full_text":        full_text,
    }


# ============================================================
# 4. 고객 평판
# ============================================================

def get_score_label(score: float) -> str:
    if score >= 4.5:   return "높음"
    elif score >= 4.0: return "보통"
    else:              return "낮음"


def get_diff_str(diff: float) -> str:
    if abs(diff) < 0.05:
        return "동종업계 평균과 동일한 수준"
    elif diff > 0:
        return f"동종업계 평균보다 {diff:.1f}점 높은 수준"
    else:
        return f"동종업계 평균보다 {abs(diff):.1f}점 낮은 수준"


def get_reputation_text(row, full_df: pd.DataFrame) -> dict:
    latest_df = full_df[full_df["ym_quarter"] == full_df["ym_quarter"].max()]
    group     = latest_df[(latest_df["big_ind"] == row["big_ind"]) & (latest_df["dong"] == row["dong"])]
    n         = len(group)

    means = {
        "종합":   group["score"].mean(),
        "맛":     group["score_taste"].mean(),
        "가격":   group["score_price"].mean(),
        "서비스": group["score_service"].mean(),
    }

    score         = row["score"]
    score_taste   = row["score_taste"]
    score_price   = row["score_price"]
    score_service = row["score_service"]

    total_label   = get_score_label(score)
    taste_label   = get_score_label(score_taste)
    price_label   = get_score_label(score_price)
    service_label = get_score_label(score_service)

    sub_labels = {"맛": taste_label, "가격": price_label, "서비스": service_label}
    low_items  = [k for k, v in sub_labels.items() if v == "낮음"]
    high_items = [k for k, v in sub_labels.items() if v == "높음"]

    if len(low_items) >= 2:
        highlight = f"{'·'.join(low_items)} 만족도가 전반적으로 낮은 점포입니다."
    elif len(low_items) == 1:
        highlight = f"{low_items[0]} 만족도가 낮은 점포입니다."
    elif high_items and not low_items:
        highlight = f"{'·'.join(high_items)} 만족도가 높은 점포입니다."
    else:
        highlight = "전반적으로 평균 수준의 점포입니다."

    total_text = (
        f"{get_diff_str(score - means['종합'])}으로, {total_label}입니다. ({score:.1f}점 / 5점 만점)"
    )

    detail_lines = []
    for name, label, val, key in [
        ("맛",     taste_label,   score_taste,   "맛"),
        ("가격",   price_label,   score_price,   "가격"),
        ("서비스", service_label, score_service, "서비스"),
    ]:
        detail_lines.append(
            f"{name} 만족도는 {label}으로, {get_diff_str(val - means[key])}입니다. ({val:.1f}점)"
        )

    full_text = "\n".join([
        "[고객 평판]",
        f"종합 진단: {highlight}", "",
        "1. 종합 평점", total_text, "",
        "2. 세부 항목", "\n".join(detail_lines),
    ])

    return {
        "current_text":  full_text,
        "total_label":   total_label,
        "taste_label":   taste_label,
        "price_label":   price_label,
        "service_label": service_label,
        "highlight":     highlight,
        "means":         means,
        "n":             n,
    }


# ============================================================
# 5. 위험 진단
# ============================================================

CUST_PRED_MAP = {
    "1_Growth": "성장형 — 유니크 고객 수가 많고 재방문 비중도 높아 성장 가능성이 높은 고객군",
    "2_Loyal":  "단골형 — 유니크 고객 수는 적지만 재방문 비중이 높아 충성도 중심 고객군",
    "3_Trial":  "체험형 — 유니크 고객 수는 많지만 재방문 비중이 낮아 체험 중심 고객군",
    "4_AtRisk": "위기형 — 유니크 고객 수가 적고 재방문 비중도 낮아 관리가 필요한 위험 고객군",
}

MKT_PRED_MAP = {
    3.0: "활성화·성장 상권 — 신규 개업이 활발하고 성장 잠재력이 큰 상권",
    4.0: "일시적 변동 상권 — 개폐업 변동성이 있어 예측 불가능한 위험 요인이 존재하는 상권",
    2.0: "성숙·쇠퇴 상권 — 성장이 멈추고 쇠퇴기에 진입한 포화 상권",
    1.0: "혼란·경쟁 상권 — 진입과 이탈이 반복되며 경쟁이 치열하고 변동성이 높은 상권",
}


def get_score_level(score: float) -> str:
    if score <= 33:   return "안정"
    elif score <= 66: return "주의"
    else:             return "위험"


def get_risk_combo(sales_level: str, cust_level: str, mkt_level: str) -> str | None:
    risk_count = sum(1 for l in [sales_level, cust_level, mkt_level] if l == "위험")
    warn_count = sum(1 for l in [sales_level, cust_level, mkt_level] if l in ("위험", "주의"))

    if risk_count == 3:
        return "매출·고객·상권 전 영역 위험 신호 — 즉각적인 대응이 필요한 상태"
    elif risk_count == 2:
        areas = [n for n, l in [("매출", sales_level), ("고객", cust_level), ("상권", mkt_level)] if l == "위험"]
        return f"{' · '.join(areas)} 위험 신호 — 복합 위험 상태"
    elif risk_count == 1:
        area = next(n for n, l in [("매출", sales_level), ("고객", cust_level), ("상권", mkt_level)] if l == "위험")
        return f"{area} 영역 위험 — 집중 관리 필요"
    elif warn_count >= 2:
        return "복수 영역 주의 단계 — 지속 모니터링 필요"
    return None


def get_peer_risk_rank(row, full_df: pd.DataFrame) -> str:
    latest_df = full_df[full_df["ym_quarter"] == full_df["ym_quarter"].max()]
    group     = latest_df[latest_df["big_ind"] == row["big_ind"]].copy()
    n         = len(group)

    group = group.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    rank  = group[group["id"] == row["id"]].index[0] + 1
    tier  = rank_to_tier(rank, n)

    mean_score = group["final_risk_score"].mean()
    diff       = row["final_risk_score"] - mean_score
    direction  = "높은 수준" if diff > 0 else "낮은 수준"

    return (
        f"동일 업종 {n}개 점포 중 {rank}위로 {tier}에 속하며, "
        f"업종 평균보다 위험 점수가 {abs(diff):.1f}점 더 {direction}입니다."
    )


def get_risk_text(row, full_df: pd.DataFrame) -> dict:
    sales_score = row["sales_score"]
    cust_score  = row["cust_score"]
    mkt_score   = row["mkt_score"]
    final_score = row["final_risk_score"]

    sales_level = get_score_level(sales_score)
    cust_level  = get_score_level(cust_score)
    mkt_level   = get_score_level(mkt_score)
    final_level = get_score_level(final_score)

    combo     = get_risk_combo(sales_level, cust_level, mkt_level)
    peer_rank = get_peer_risk_rank(row, full_df)

    cust_pred_text = CUST_PRED_MAP.get(row["cust_pred"], row["cust_pred"])
    mkt_pred_text  = MKT_PRED_MAP.get(row["mkt_pred"],  row["mkt_pred"])

    diagnosis = f"{combo}." if combo else "전반적으로 안정적인 상태입니다."

    full_text = "\n".join([
        "[위험 진단]",
        f"종합 진단: {diagnosis}", "",
        "1. 예상 고객 유형", cust_pred_text, "",
        "2. 예상 상권 유형", mkt_pred_text, "",
        "3. 위험 점수",
        f"매출 위험도 {sales_level} ({sales_score:.1f}점)",
        f"고객 위험도 {cust_level} ({cust_score:.1f}점)",
        f"상권 위험도 {mkt_level} ({mkt_score:.1f}점)",
        f"종합 위험도 {final_level} ({final_score:.1f}점) — {peer_rank}",
    ])

    return {
        "current_text":   full_text,
        "final_level":    final_level,
        "sales_level":    sales_level,
        "cust_level":     cust_level,
        "mkt_level":      mkt_level,
        "combo":          combo,
        "cust_pred_text": cust_pred_text,
        "mkt_pred_text":  mkt_pred_text,
        "peer_rank":      peer_rank,
    }


# ============================================================
# 6. 최종 컨텍스트 조합
# ============================================================

def build_context(store_name: str, df: pd.DataFrame) -> str:
    """
    가맹점명으로 전체 분석 컨텍스트 문자열 생성
    Args:
        store_name: 가맹점 title (예: "성수AGU")
        df        : score_review_merged.csv 로드한 DataFrame
    Returns:
        LLM에 전달할 컨텍스트 문자열
    """
    store_df = df[df["title"] == store_name]
    if store_df.empty:
        raise ValueError(f"'{store_name}' 가게를 찾을 수 없습니다.")

    latest_quarter = store_df["ym_quarter"].max()
    row      = store_df[store_df["ym_quarter"] == latest_quarter].iloc[0]
    store_id = row["id"]

    store_all          = df[df["id"] == store_id].sort_values("ym_quarter")
    sales_amt_quarters = store_all["sales_amt_cat_mean"].tolist()
    sales_cnt_quarters = store_all["sales_cnt_cat_mean"].tolist()
    aov_quarters       = store_all["aov_cat_mean"].tolist()

    sales_result      = sales_to_text(sales_amt_quarters, sales_cnt_quarters, aov_quarters, store_id, df)
    customer_result   = customer_to_text(row["re_cust_rat_mean"], row["new_cust_rat_mean"], row, df)
    district_result   = district_to_text(row, df)
    reputation_result = get_reputation_text(row, df)
    risk_result       = get_risk_text(row, df)

    basic_info = f"업종: {row['big_ind']}\n세부카테고리: {row['category']}\n위치: {row['dong']}"

    return "\n\n".join([
        f"=== 가맹점명: {store_name} ===",
        f"[기본 정보]\n{basic_info}",
        risk_result["current_text"],
        sales_result["full_text"],
        customer_result["full_text"],
        district_result["full_text"],
        reputation_result["current_text"],
    ])