# streamlit run app.py

import streamlit as st
import pandas as pd
from typing import Set

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data():
    """CSV 파일을 로드하고 키워드 리스트를 생성합니다."""
    try:
        df_products = pd.read_csv('products.csv')
        df_influencers = pd.read_csv('influencers.csv')
        df_results = pd.read_csv('campaign_results.csv') # NEW: 성과 데이터 로드
        
        # 키워드 문자열을 리스트/셋으로 변환하는 함수
        def keyword_to_set(keyword_str):
            if isinstance(keyword_str, str):
                return set(keyword_str.replace(' ', '').split(','))
            return set()

        # 키워드/태그 셋 컬럼 추가
        df_products['텍스트_키워드_SET'] = df_products['핵심_성분/키워드'].apply(keyword_to_set)
        df_products['시각_키워드_SET'] = df_products['브랜드_이미지_태그'].apply(keyword_to_set) 
        
        df_influencers['텍스트_키워드_SET'] = df_influencers['주요_콘텐츠_키워드'].apply(keyword_to_set)
        df_influencers['시각_키워드_SET'] = df_influencers['평균_피드_감성_태그'].apply(keyword_to_set) 

        # 성과 데이터와 인플루언서 데이터 병합 (분석 편의성 위해)
        df_influencers_with_results = df_influencers.merge(
            df_results.groupby('Influencer_ID').agg(
                캠페인_참여_횟수=('Campaign_ID', 'count'),
                평균_전환율=('전환율', 'mean'),
                평균_긍정_감정비율=('긍정_감정비율', 'mean')
            ).reset_index(),
            left_on='ID',
            right_on='Influencer_ID',
            how='left'
        ).fillna(0) # 참여하지 않은 인플루언서는 0으로 처리

        return df_products, df_influencers_with_results
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 'data_generator.py'를 실행하여 파일을 먼저 생성해 주세요.")
        return pd.DataFrame(), pd.DataFrame()

# --- 핵심 로직: 자카드 유사도 계산 (유지) ---
def calculate_jaccard_similarity(set1, set2):
    """두 키워드/태그 집합 간의 자카드 유사도를 계산합니다."""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# --- LLM 시뮬레이션 함수 (유지) ---
def simulate_llm_analysis(prompt: str, df_products: pd.DataFrame) -> tuple[Set[str], Set[str], str]:
    """
    마케터의 프롬프트(요청)를 분석하여 가상의 키워드와 태그를 추출합니다.
    """
    prompt_lower = prompt.lower()
    
    selected_product = None
    
    if '수분' in prompt_lower or '저자극' in prompt_lower:
        selected_product = df_products[df_products['제품명'] == '수분_앰플'].iloc[0]
    elif '미백' in prompt_lower or '비타민' in prompt_lower:
        selected_product = df_products[df_products['제품명'] == '미백_세럼'].iloc[0]
    elif '진정' in prompt_lower or '시카' in prompt_lower:
        selected_product = df_products[df_products['제품명'] == '진정_크림'].iloc[0]
    elif '모공' in prompt_lower or '피지' in prompt_lower:
        selected_product = df_products[df_products['제품명'] == '모공_클렌저'].iloc[0]
    
    if selected_product is not None:
        text_keywords = selected_product['텍스트_키워드_SET']
        visual_tags = selected_product['시각_키워드_SET']
        product_name = selected_product['제품명']
    else:
        text_keywords = {'비건', '민감성', '트렌디'}
        visual_tags = {'미니멀', '저채도'}
        product_name = "신규 캠페인"


    llm_summary = f"""**[LLM 분석 요약]**
    마케터님의 요청 '{prompt[:50]}...'을 분석한 결과, **'{product_name}'** 캠페인에 가장 적합한
    **핵심 키워드**와 **시각적 감성 태그**를 추출했습니다. 
    이 파라미터를 기반으로 인플루언서 매칭을 진행합니다.
    """
    return text_keywords, visual_tags, llm_summary

# --- 모듈별 함수 정의 ---

def matching_module(df_products, df_influencers):
    """LLM 기반 적합도 매칭 모듈"""
    st.title("🧠 생성형 AI 기반 맞춤형 인플루언서 매칭")
    st.markdown("마케팅 요청을 자연어로 입력하면, AI가 자동으로 키워드와 감성을 추출하여 인플루언서를 추천합니다.")
    st.markdown("---")
    
    # 1. 사이드바: LLM 입력
    st.sidebar.header("1. 마케팅 요구사항 입력 (LLM Prompt)")
    prompt = st.sidebar.text_area(
        "원하는 제품/브랜드 이미지에 대해 설명하세요:",
        value="요즘 MZ 세대를 타겟으로, 수분 보충이 확실하고 인스타 감성이 잘 맞는 시크한 무드의 마이크로 인플루언서를 추천해줘.",
        height=150
    )
    
    if not prompt:
        st.warning("마케팅 요구사항을 입력해 주세요.")
        return

    # 2. LLM 시뮬레이션 실행
    text_keywords_set, visual_tags_set, llm_summary = simulate_llm_analysis(prompt, df_products)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. LLM 분석 결과")
    st.sidebar.markdown(llm_summary)
    st.sidebar.markdown(f"**텍스트 키워드:** `{', '.join(text_keywords_set)}`")
    st.sidebar.markdown(f"**시각적 감성 태그:** `{', '.join(visual_tags_set)}`")
    st.sidebar.markdown("---")

    # 3. 매칭 가중치 설정
    st.sidebar.subheader("3. 매칭 가중치 설정")
    w_text = st.sidebar.slider("텍스트 키워드 중요도 (W_텍스트)", 0.0, 1.0, 0.6, 0.05)
    w_visual = 1.0 - w_text
    st.sidebar.info(f"시각적 감성 중요도 (W_시각): **{w_visual:.2f}**")
    st.sidebar.markdown("---")

    # 4. 적합도 계산 및 결과 테이블 생성
    results = []
    for index, influencer in df_influencers.iterrows():
        # 1. 텍스트 적합도
        text_score = calculate_jaccard_similarity(
            text_keywords_set,
            influencer['텍스트_키워드_SET']
        ) * 100
        
        # 2. 시각적 적합도
        visual_score = calculate_jaccard_similarity(
            visual_tags_set,
            influencer['시각_키워드_SET']
        ) * 100
        
        # 3. 최종 종합 점수 (가중 평균)
        final_matching_score = (w_text * text_score) + (w_visual * visual_score)
        
        # 일치 항목 추출
        matched_text_keywords = text_keywords_set.intersection(influencer['텍스트_키워드_SET'])
        matched_visual_tags = visual_tags_set.intersection(influencer['시각_키워드_SET'])

        # 참여율 (ER) 계산
        er = ((influencer['평균_좋아요'] + influencer['평균_댓글']) / influencer['팔로워 수']) * 100
        
        results.append({
            '이름': influencer['이름'],
            '플랫폼': influencer['플랫폼'],
            '팔로워 수': influencer['팔로워 수'],
            '참여율 (ER)': f"{er:.2f}%",
            '최종_종합_점수': final_matching_score,
            '텍스트_적합도': text_score,
            '시각적_적합도': visual_score,
            '일치_키워드': ', '.join(matched_text_keywords) if matched_text_keywords else '없음',
            '일치_감성_태그': ', '.join(matched_visual_tags) if matched_visual_tags else '없음',
        })

    df_results = pd.DataFrame(results)
    
    # 5. 결과 시각화
    df_results_sorted = df_results.sort_values(by='최종_종합_점수', ascending=False)
    
    st.header(f"✨ AI 매칭 결과: 인플루언서 추천 리스트")
    
    top_5 = df_results_sorted.head(5).copy()
    top_5.index = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    
    st.subheader("⭐ 추천 인플루언서 (Top 5)")
    st.dataframe(
        top_5[['이름', '최종_종합_점수', '텍스트_적합도', '시각적_적합도', '참여율 (ER)', '일치_키워드', '일치_감성_태그']],
        column_config={
            "최종_종합_점수": st.column_config.ProgressColumn("최종 점수", format="%.1f점", min_value=0, max_value=100, help="텍스트와 시각적 적합도의 가중 평균"),
            "텍스트_적합도": st.column_config.NumberColumn("텍스트 적합도", format="%.1f점"),
            "시각적_적합도": st.column_config.NumberColumn("시각적 적합도", format="%.1f점"),
            "참여율 (ER)": st.column_config.TextColumn("참여율 (ER)"),
        },
        use_container_width=True
    )

def portfolio_module(df_influencers):
    """Win-Win 협업 제안 (데이터 포트폴리오) 모듈"""
    st.title("🤝 Win-Win 협업 제안: 인플루언서 데이터 포트폴리오")
    st.markdown("단순 팔로워 수가 아닌, 데이터로 증명된 **진정한 영향력**을 보여줌으로써 투명한 협업 관계를 구축합니다.")
    st.markdown("---")

    # 1. 인플루언서 선택
    influencer_names = df_influencers['이름'].unique().tolist()
    selected_influencer_name = st.selectbox(
        "데이터 포트폴리오를 확인할 인플루언서를 선택하세요:",
        influencer_names
    )

    selected_data = df_influencers[df_influencers['이름'] == selected_influencer_name].iloc[0]
    
    # 포트폴리오 헤더
    st.subheader(f"✨ {selected_influencer_name} 님의 협업 가치 리포트")

    st.markdown("""
        > **핵심 메시지:** 이 리포트는 **당신의 진정성, 참여율, 실제 전환 기여도**를 객관적으로 증명하여, 
        > 향후 브랜드와의 협상력을 높이는 포트폴리오 자료로 활용될 수 있습니다. 
        > 우리는 당신의 진정한 영향력에 투자합니다.
    """)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    
    # 2. 핵심 지표 시각화 (Engagement & Authenticity)
    er = ((selected_data['평균_좋아요'] + selected_data['평균_댓글']) / selected_data['팔로워 수']) * 100
    
    col1.metric("팔로워 수", f"{selected_data['팔로워 수']:,} 명")
    col2.metric("평균 참여율 (ER)", f"{er:.2f}%", help="팔로워 대비 좋아요/댓글 수로 계산된 활동성 지표입니다.")
    col3.metric("진정성 지수", f"{selected_data['진정성_지수']:.0f} 점", help="AI 봇 활동 탐지 및 진정성 분석을 통해 산출된 지표입니다.")
    col4.metric("캠페인 참여 횟수", f"{selected_data['캠페인_참여_횟수']:.0f} 회")

    st.markdown("---")

    # 3. 비즈니스 성과 (전환 및 감성 분석)
    st.subheader("📊 비즈니스 기여도 분석 (Campaign Performance)")
    
    col5, col6, col7 = st.columns(3)
    
    # 참여 이력이 있는 경우에만 표시
    if selected_data['캠페인_참여_횟수'] > 0:
        col5.metric(
            "평균 전환율 (Conversion Rate)", 
            f"{selected_data['평균_전환율']:.2f} %",
            help="캠페인 클릭 대비 실제 구매로 이어진 비율의 평균입니다."
        )
        col6.metric(
            "평균 긍정 감성 기여", 
            f"{selected_data['평균_긍정_감정비율']:.1f} %",
            help="콘텐츠 리뷰 및 댓글에 대한 AI 감성 분석 결과, 긍정 반응이 나타난 비율의 평균입니다."
        )
        col7.markdown("**성과 기반 인센티브 예상 (가상)**")
        col7.markdown(f"**💰 {selected_data['평균_전환율'] * 10000:.0f} 원**")

    else:
        st.info("이 인플루언서의 캠페인 성과 데이터는 아직 없습니다. 다음 협업으로 첫 데이터 포트폴리오를 만들어 보세요!")

    st.markdown("---")

    # 4. 콘텐츠 적합성 분석
    st.subheader("💡 콘텐츠 적합성 (Niche & Alignment)")
    col8, col9 = st.columns(2)
    col8.markdown("**주요 콘텐츠 키워드 (전문 분야)**")
    col8.code(selected_data['주요_콘텐츠_키워드'])
    
    col9.markdown("**평균 피드 감성 (Tone & Manner)**")
    col9.code(selected_data['평균_피드_감성_태그'])


# --- Streamlit 앱 정의 ---
def app():
    st.set_page_config(
        page_title="K-Beauty 인플루언서 마케팅 대시보드",
        layout="wide"
    )

    df_products, df_influencers = load_data()
    
    if df_products.empty or df_influencers.empty:
        return

    # 탭 구현
    tab1, tab2 = st.tabs(["🎯 LLM 기반 적합도 매칭", "🤝 Win-Win 포트폴리오 제안"])
    
    with tab1:
        matching_module(df_products, df_influencers)
    
    with tab2:
        portfolio_module(df_influencers)

# 앱 실행
if __name__ == "__main__":
    app()