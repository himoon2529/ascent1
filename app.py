import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================
# 1. 소비자 유형별 기준 데이터
# ==================================

criteria = {
    "전통주의자": [
        "안전성, 신뢰감을 중시",
        "품질에 관한 정보 중요시",
        "근검절약",
        "새로운 것에 개방적이지 않다",
        "세부사항을 오래 살펴본다",
        "모든 것을 꼼꼼히 검증",
        "수동적,전통적",
        "우뇌 발달"
    ],

    "조화론자": [
        "가정의 안정감, 화합을 중요시한다",
        "수동적,느긋함",
        "행동을 야기하는 핵심 요인: 결합, 돌봄 모듈",
        "사회호르몬이자 애정호르몬인 옥시토신 농도가 짙다",
        "우뇌 발달",
        "여성의 비중이 훨씬 높다",
        "전통주의자에 비해 타인에게 개방적이다",
        "정원, 가정, 반려동물 관련 제품을 선호하며 이에 지대한 관심을 보인다"
    ],

    "개방주의자": [
        "양뇌 발달",
        "개방적, 긍정적인 생활방식을 지니고 있다",
        "제품의 향유가능성을 매우 중요시한다",
        "누리고, 즐기는 것을 중요시함",
        "타인과의 접촉을 중시한다",
        "문화공연, 이벤트, 가족과 함께하는 체험을 좋아한다",
        "품질, 원료, 건강에도 어느 정도는 관심이 있음",
        "행복한 체험을 약속하는 상품을 선호한다"
    ],

    "쾌락주의자": [
        "자극 시스템, 도파민이 주도권",
        "품질,원산지를 중요하게 여기지 않음",
        "최고의 가치는 색다른 것, 새로운 것에 있음",
        "충동구매자",
        "유행에 민감",
        "좌뇌 발달",
        "즐거움과 쾌락을 중요시",
        "심사숙고와는 거리가 멀다"
    ],

    "모험가": [
        "도파민, 테스토스테론 농도 짙음",
        "쾌락주의자적 요소에 전투적, 충동적 요소가 더욱 더해짐",
        "가시적으로 드러나는 뛰어난 성능, 즐거움",
        "구매처 충성도, 건강, 관습, 품질에는 별로 관심 없음",
        "유행에 민감",
        "좌뇌 발달",
        "성능이 개선되고 자유를 선사할 수 있는 제품을 선호한다",
        "특가 행사를 좋아한다"
    ],

    "실행가": [
        "테스토스테론 농도 짙음",
        "자신을 과시하는 것을 좋아함",
        "세일을 좋아하는 등 영리한 소비를 추구한다",
        "영리한 소비를 추구하지만, 주변 사람들 눈에 띄는 제품들은 과시가능한 제품을 선호한다",
        "타인에게 노출되는 제품은 저가로 거의 사지 않는다",
        "과시적 제품이 아니라면, 가격 흥정에 적극적이다",
        "해당 제품을 구매해 지위와 인기의 특권이 보장된다면 가격을 신경쓰지 않는다",
        "좌뇌 발달"
    ],

    "규율숭배자": [
        "우뇌 발달",
        "절약정신이 많다",
        "예상하지 못한 사건을 싫어한다",
        "품질, 보증 관련 요소를 중요시한다",
        "제품의 가격을 충분히 비교하고 소비하는 편이다",
        "여러 요소를 고려하여 구매하기 때문에 구매결정까지의 시간이 오래 걸린다",
        "유행을 선호하지 않고, 복잡한 것을 싫어한다",
        "제품의 최고의 가치는 기능적 측면에 있다고 본다"
    ]
}


# ==================================
# 2. AI 모델 불러오기
# ==================================

@st.cache_resource
def load_resources():

    model = SentenceTransformer(
        "jhgan/ko-sroberta-multitask"
    )

    reference_vectors = {}

    for name, sentences in criteria.items():

        vectors = model.encode(
            sentences,
            normalize_embeddings=True
        )

        reference_vectors[name] = vectors

    return model, reference_vectors


# ==================================
# 3. 상품 특성 분석 함수
# ==================================

def analyze_product(product_features, model, reference_vectors):

    product_vectors = model.encode(
        product_features,
        normalize_embeddings=True
    )

    results = []

    for name, ref_vectors in reference_vectors.items():

        similarities = cosine_similarity(
            product_vectors,
            ref_vectors
        )

        best_similarities = similarities.max(axis=1)

        score = best_similarities.mean()

        results.append({
            "소비자 유형": name,
            "유사도 점수": round(float(score), 4)
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="유사도 점수",
        ascending=False
    ).reset_index(drop=True)

    result_df.insert(
        0,
        "순위",
        range(1, len(result_df) + 1)
    )

    return result_df


# ==================================
# 4. 웹사이트 화면 설정
# ==================================

st.set_page_config(
    page_title="소비자 유형 분석 시스템",
    page_icon="📊",
    layout="centered"
)

st.title("소비자 유형 분석 시스템")

st.caption(
    "림빅 모델 기반 소비자 유형 분류 알고리즘"
)

st.divider()

st.subheader("상품 특성 입력")

st.write(
    "분석하고 싶은 상품의 특징을 한 줄에 하나씩 입력하세요."
)

product_input = st.text_area(
    "상품 특성",
    placeholder="예시:\n독창적인 디자인을 적용하였다.\n사용자의 편리성을 높이는 기능을 제공한다.\n오랫동안 사용할 수 있는 내구성을 갖추었다.",
    height=180
)


# ==================================
# 5. 분석 버튼 및 결과 출력
# ==================================

if st.button("분석하기", use_container_width=True):

    product_features = [
        line.strip()
        for line in product_input.splitlines()
        if line.strip()
    ]

    if not product_features:

        st.warning("상품 특성을 먼저 입력해 주세요.")

    else:

        with st.spinner("소비자 유형을 분석하고 있습니다..."):

    try:

                model, reference_vectors = load_resources()

                result = analyze_product(
                    product_features,
                    model,
                    reference_vectors
                )

                top_type = result.iloc[0]["소비자 유형"]

                st.divider()

                st.subheader("소비자 유형 분석 결과")

                st.success(
                    f"상대적 관련성이 가장 높은 유형: {top_type}"
                )

                st.subheader("유형별 분석 순위")

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    result.set_index("소비자 유형")["유사도 점수"]
                )

                st.caption(
                    "※ 유사도 점수는 상품 특성과 유형별 기준 문장 간의 "
                    "상대적 의미 유사도이며 실제 소비자 비율이나 "
                    "유형 판별 확률을 의미하지 않습니다."
                )

    except Exception as e:
     st.error("분석 중 오류가 발생했습니다.")
     st.exception(e)
