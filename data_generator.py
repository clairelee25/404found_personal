import pandas as pd
import random
import numpy as np

# 가상 데이터 셋업
random.seed(42)

# 1. 제품 데이터 (Products)
product_keywords = {
    '수분_앰플': (['히알루론산', '수분', '속건조', '저자극', '비건'], ['청순,내추럴', '미니멀,시크', '저채도,웜톤']),
    '미백_세럼': (['비타민C', '미백', '잡티', '브라이트닝', '기능성'], ['강렬,비비드', '고화질,깔끔', '채도높음,쿨톤']),
    '진정_크림': (['병풀', '시카', '진정', '민감성', '트러블'], ['자연친화,편안함', '실내,깔끔', '파스텔,웜톤']),
    '모공_클렌저': (['모공', '피지', '각질', '약산성', '클렌징'], ['힙,트렌디', '실내,깔끔', '모노톤,무채색']),
}

products_data = []
for p_id, (p_name, (keywords, visual_tags)) in enumerate(product_keywords.items(), 101):
    products_data.append({
        'Product_ID': p_id,
        '제품명': p_name,
        '카테고리': p_name.split('_')[1],
        '핵심_성분/키워드': ','.join(keywords),
        '소비자_니즈_키워드': ','.join(keywords[:3]),
        '브랜드_이미지_태그': ','.join(visual_tags)
    })

df_products = pd.DataFrame(products_data)
df_products.to_csv('products.csv', index=False, encoding='utf-8')

# 2. 인플루언서 데이터 (Influencers)
influencer_data = []
all_keywords = list(set(k for keywords, visual_tags in product_keywords.values() for k in keywords)) + ['화장품하울', '내돈내산', '뷰티팁', '메이크업']
all_visual_tags = ['밝은색감,인물중심', '야외,풍경', '실내,깔끔', '필름감성,저채도', '파스텔,웜톤', '모노톤,무채색', '채도높음,쿨톤']


for i in range(1, 21): # 20명의 가상 인플루언서
    follower = random.randint(1000, 100000)
    content_keywords = random.sample(all_keywords, k=random.randint(2, 5))
    
    if follower <= 10000:
        authenticity = random.randint(80, 95)
    else:
        authenticity = random.randint(50, 85)

    influencer_data.append({
        'ID': i,
        '이름': f'인플루언서_{i}',
        '플랫폼': random.choice(['TikTok', 'Instagram', 'YouTube']),
        '팔로워 수': follower,
        '평균_좋아요': int(follower * random.uniform(0.05, 0.15)),
        '평균_댓글': int(follower * random.uniform(0.005, 0.02)),
        '평균_공유': int(follower * random.uniform(0.003, 0.01)),
        '평균_저장': int(follower * random.uniform(0.003, 0.01)),
        '진정성_지수': authenticity, # 0~100 스케일
        '주요_콘텐츠_키워드': ','.join(content_keywords),
        '평균_피드_감성_태그': random.choice(all_visual_tags),
    })

df_influencers = pd.DataFrame(influencer_data)
df_influencers.to_csv('influencers.csv', index=False, encoding='utf-8')

# 3. 캠페인 성과 데이터 (Campaign_Results)
campaign_results = []
campaign_names = ['Summer_Hydration_2024', 'Brightening_Winter_Sale', 'Acne_SOS_Launch']
all_influencer_ids = df_influencers['ID'].tolist()

for c_id, campaign in enumerate(campaign_names, 1):
    # 각 캠페인에 참여한 가상 인플루언서 8명 랜덤 선택
    participants = random.sample(all_influencer_ids, 8) 
    
    for inf_id in participants:
        inf_data = df_influencers[df_influencers['ID'] == inf_id].iloc[0]
        
        # 팔로워 규모에 따라 노출/전환율 다르게 설정 (나노/마이크로의 고효율 시뮬레이션)
        is_nano_micro = inf_data['팔로워 수'] <= 30000
        
        exposure = int(inf_data['팔로워 수'] * random.uniform(0.5, 1.5))
        conversion_rate = np.clip(random.gauss(0.8 if is_nano_micro else 0.4, 0.3), 0.1, 1.5) # 전환율 %
        
        # AI 감정 분석 시뮬레이션
        if is_nano_micro and inf_data['진정성_지수'] > 80:
             positive_sentiment = random.uniform(70, 95)
        else:
             positive_sentiment = random.uniform(40, 75)
             
        negative_sentiment = 100 - positive_sentiment - random.uniform(0, 5) # 중립 감성 가정
        
        campaign_results.append({
            'Campaign_ID': campaign,
            'Influencer_ID': inf_id,
            '제품명': random.choice(df_products['제품명'].tolist()),
            '노출수': exposure,
            '전환율': conversion_rate,
            '긍정_감정비율': positive_sentiment,
            '부정_감정비율': negative_sentiment,
            'AI_분석_요약': random.choice(['매우 높은 진정성과 전환 기여도를 보임.', '일반적인 노출 효과, 감성 반응은 중간 수준.', '노출 대비 참여율이 낮음. 콘텐츠 내용 보완 필요.'])
        })

df_campaign_results = pd.DataFrame(campaign_results)
df_campaign_results.to_csv('campaign_results.csv', index=False, encoding='utf-8')

print("products.csv, influencers.csv, campaign_results.csv 파일이 재생성되었습니다.")