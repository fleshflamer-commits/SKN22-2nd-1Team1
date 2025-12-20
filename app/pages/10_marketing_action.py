import streamlit as st
import pandas as pd
import sys
import os
import requests
from io import BytesIO
from ui.header import render_header

# --- [STEP 1] 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(app_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# --- [STEP 2] 모듈 임포트 ---
try:
    from service.CustomerCareCenter import PurchaseIntentService 
    from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter
except ImportError:
    from app.service.CustomerCareCenter import PurchaseIntentService
    from app.adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter

# --- [STEP 3] 데이터 및 서비스 로드 ---
@st.cache_resource
def init_service():
    model_path = "artifacts/best_pr_auc_balancedrf.joblib"
    adapter = PurchaseIntentPRAUCModelAdapter(model_path) 
    return PurchaseIntentService(adapter), adapter

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/test.csv")

service, adapter = init_service()

# =========================================================
# [수정] 기존 10개 → 30개 세션으로 확장
# =========================================================
df = load_data().head(30)
# =========================================================

# =========================================================
# [유지] 그룹별 Google Drive 이미지 (총 10개)
# =========================================================
GROUP_IMAGE_MAP = {
    1: "https://drive.google.com/uc?id=1C_FatsRQqIQPnygwx3McU4NcZZiRiBFp",
    2: "https://drive.google.com/uc?id=1KB0J8zrm7ZFC4FvAcL3Z1r831ZBZpKJR",
    3: "https://drive.google.com/uc?id=1n7l5AhZIU46u7vD6UxBk7xWSaraVMDao",
    4: "https://drive.google.com/uc?id=1tmKyCJ_qhVv0050H9DPNdckhgXV0QwBt",
    5: "https://drive.google.com/uc?id=1o3XvxRP9-iN80cO8T_aVPoA04CMk94hh",
    6: "https://drive.google.com/uc?id=1QUXQMxvR0b7Gyx-KsHidAA6kVg8nFp_Y",
    7: "https://drive.google.com/uc?id=1yJ5An-fs3J8PlADZ4NYUp4ySiFh4Okct",
    8: "https://drive.google.com/uc?id=1u7eZMaBMpQ2aqg5A9BRw5BP1eKu6Kw4m",
    9: "https://drive.google.com/uc?id=1kU9k2cCKkHRNnHhLCYKy4QsIQdTYAyCc",
    10:"https://drive.google.com/uc?id=1kZpn2fKK2yC1PImdHo2CwQ61DVWf9qSy",
}

@st.cache_data
def load_image_from_drive(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    return None

# --- UI ---
render_header()
st.title("🎯 마케팅 전략 가이드 시뮬레이터")

# =========================================================
# [수정] 안내 문구도 30개 기준으로 변경
# =========================================================
st.info("💡 분석 효율을 위해 상위 30개의 주요 타겟 세션을 요약하여 제공합니다.")
# =========================================================

left_col, right_col = st.columns([4, 6])

with left_col:
    st.subheader("📥 타겟 세션 리스트 (TOP 30)")

    # =========================================================
    # [수정] 세션 선택 리스트 30개로 확장
    # =========================================================
    labels = [f"세션 분석 대상 #{i+1}" for i in range(len(df))]
    selected_label = st.selectbox("분석할 세션 선택", labels)

    group_id = labels.index(selected_label) + 1
    row = df.iloc[group_id - 1]
    # =========================================================

    X_one = pd.DataFrame([row.drop("Revenue", errors="ignore")])
    proba = float(adapter.predict_proba(X_one).iloc[0])
    risk = service.classify_risk(proba)
    action = service.recommend_action(row.to_dict(), proba, group_id=group_id)

    st.write("---")
    st.metric("페이지 가치 (Value)", f"{row.get('PageValues', 0):.2f}")
    st.metric("이탈률 (Bounce)", f"{row.get('BounceRates', 0)*100:.1f}%")
    st.metric("체류 시간 (Duration)", f"{row.get('ProductRelated_Duration', 0):.1f}s")

with right_col:
    st.subheader("👤 고객 페르소나 진단")

    # =========================================================
    # [수정] 30개 그룹 → 이미지 10개를 순환 매핑
    # 예: 11 → 1번 이미지, 12 → 2번 이미지 …
    # =========================================================
    image_key = ((group_id - 1) % 10) + 1
    img_bytes = load_image_from_drive(GROUP_IMAGE_MAP.get(image_key))
    # =========================================================

    if img_bytes:
        st.image(img_bytes, width=620)
    else:
        st.warning("⚠️ 페르소나 이미지를 불러오지 못했습니다.")

    st.write(f"**실시간 구매 전환 확률: {proba*100:.1f}%**")
    st.progress(proba)

    if risk == "HIGH_RISK":
        status_text = "🚨 이탈 위험 높음: 즉각적인 케어가 필요합니다."
    elif risk == "OPPORTUNITY":
        status_text = "⚠️ 망설이는 단계: 혜택이 전환의 열쇠입니다."
    else:
        status_text = "✅ 구매 유력 상태: 흐름을 방해하지 마세요."

    st.markdown("### 📌 추천 마케팅 전략")
    st.markdown(f"**{status_text}**")
    st.markdown(f"> {action}")

with st.expander("🔍 선택된 세션 상세 로그 확인"):
    st.table(pd.DataFrame([row]).T)


# GROUP_IMAGE_MAP = {
#     1: "https://drive.google.com/uc?id=1C_FatsRQqIQPnygwx3McU4NcZZiRiBFp",
#     2: "https://drive.google.com/uc?id=1KB0J8zrm7ZFC4FvAcL3Z1r831ZBZpKJR",
#     3: "https://drive.google.com/uc?id=1n7l5AhZIU46u7vD6UxBk7xWSaraVMDao",
#     4: "https://drive.google.com/uc?id=1tmKyCJ_qhVv0050H9DPNdckhgXV0QwBt",
#     5: "https://drive.google.com/uc?id=1o3XvxRP9-iN80cO8T_aVPoA04CMk94hh",
#     6: "https://drive.google.com/uc?id=1QUXQMxvR0b7Gyx-KsHidAA6kVg8nFp_Y",
#     7: "https://drive.google.com/uc?id=1yJ5An-fs3J8PlADZ4NYUp4ySiFh4Okct",
#     8: "https://drive.google.com/uc?id=1u7eZMaBMpQ2aqg5A9BRw5BP1eKu6Kw4m",
#     9: "https://drive.google.com/uc?id=1kU9k2cCKkHRNnHhLCYKy4QsIQdTYAyCc",
#     10:"https://drive.google.com/uc?id=1kZpn2fKK2yC1PImdHo2CwQ61DVWf9qSy",
# }

# # 이미지 1 "🚨 [심폐소생술 시급] 고객님이 '뒤로 가기' 버튼과 썸 타는 중입니다! 혜택 한 줄 요약이랑 베스트 리뷰로 멱살 잡고 끌어와야 해요!"
# # 이미지 2. "🚪 '나 지금 나간다?'라고 온몸으로 외치는 중! 3초 안에 할인 쿠폰이나 무료배송 안 보여주면 영영 남남입니다. 빨리요!"
# # 이미지 3. "🧯 관심이라는 불씨가 생기기도 전에 로그아웃 각! 랜딩 페이지에 인기 상품이랑 신뢰 팍팍 가는 인증마크로 도배해서 눈길을 뺏으세요!"
# # 이미지 4.  "🪝 살짝 솔깃해 보이지만, 로딩 1초만 늦어도 떠날 분입니다. 복잡한 거 다 빼고 핵심 혜택만 코앞에 들이미세요!"
# # 이미지 5. "⚠️ 이 정도면 '밀당' 고수네요. 살까 말까 고민하는 게 보입니다. '오늘만 이 가격' 콤보 한 방이면 바로 넘어옵니다!"
# # 이미지 6. "👀 장바구니에 넣을까 말까 100번 고민 중! '최저가 보장'이나 '빠른 배송' 정보로 고객님의 우유부단함에 마침표를 찍어주세요!"
# # 이미지 7. "🎯 대어 낚기 직전입니다! '사람들이 이 제품 칭찬을 이렇게 많이 해요'라고 사회적 증거(후기/별점)를 마구 투척하세요!"
# # 이미지 8. "🔥 [결제 직전] 조금만 밀면 카드 슬래시! 한정판 쿠폰이나 '무료배송까지 얼마 안 남았어요'라는 멘트로 불을 지피세요!"
# # 이미지 9. "🛒 이미 마음은 결제 완료! 괜히 팝업 띄워서 방해하지 말고, 쿠폰 자동 적용해서 레드카펫 깔아드립시다. 결제 길만 걷게 하세요!"
# # 이미지 10.  "✅ [확정 전환] 이분은 숨만 쉬어도 구매하실 분입니다! 추가 영업은 사치일 뿐. 가볍게 '함께 사면 좋은 꿀템' 하나만 슥- 던져보세요."
# # 이런 상황에 맞는 사람 얼굴 이미지 10개 각각 1장씩 용량은 작게 제작해줘