import numpy as np
import pandas as pd
# 상위 경로 설정 덕분에 'adapters'에서 바로 가져올 수 있습니다.
from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter


class PurchaseIntentService:
    def __init__(self, adapter):
        self.adapter = adapter

    def score_top_k(self, features, top_k_ratio=0.05):
        features = features.drop(columns=["Revenue"], errors="ignore")
        proba = self.adapter.predict_proba(features)
        thr = float(np.quantile(proba.values, 1.0 - top_k_ratio))
        pred = (proba >= thr).astype(int)

        out = features.copy()
        out["purchase_proba"] = proba
        out["purchase_pred"] = pred
        return out

    def classify_risk(self, purchase_proba: float) -> str:
        if purchase_proba < 0.2:
            return "HIGH_RISK"
        elif purchase_proba < 0.6:
            return "OPPORTUNITY"
        else:
            return "LIKELY_BUYER"

    # =========================================================
    # [기존 유지] 확률 기반 10그룹 매핑 (bin 방식)
    # =========================================================
    def classify_group_10(self, purchase_proba: float) -> tuple[int, str]:
        p = float(purchase_proba)

        # HIGH_RISK: p < 0.2  -> 5 bins (0.04 each)
        if p < 0.2:
            width = 0.2 / 5  # 0.04
            group_id = int(p // width) + 1
            group_id = min(max(group_id, 1), 5)
            return group_id, f"그룹{group_id}(고위험 이탈군)"

        # OPPORTUNITY: 0.2 <= p < 0.6 -> 3 bins (~0.1333 each)
        if p < 0.6:
            width = 0.4 / 3
            group_id = int((p - 0.2) // width) + 6  # 6~8
            group_id = min(max(group_id, 6), 8)
            return group_id, f"그룹{group_id}(전환 기회군)"

        # LIKELY_BUYER: p >= 0.6 -> 2 bins (0.2 each)
        width = 0.4 / 2  # 0.2
        group_id = int((p - 0.6) // width) + 9  # 9~10
        group_id = min(max(group_id, 9), 10)
        return group_id, f"그룹{group_id}(구매 유력군)"

    def recommend_action(self, row: dict, purchase_proba: float, group_id: int | None = None) -> str:
        """
        [수정] group_id(그룹 아이디)를 선택적으로 받아서
        Streamlit의 '그룹1~10'과 메시지(10종)를 1:1로 맞출 수 있게 함.
        - group_id가 None이면: 기존처럼 확률 기반 bin 방식(classify_group_10)
        - group_id가 있으면: UI 그룹번호를 그대로 사용
        """
        bounce = row.get("BounceRates", 0)
        exit_r = row.get("ExitRates", 0)
        page_val = row.get("PageValues", 0)
        duration = row.get("ProductRelated_Duration", 0)

        # =========================================================
        # [수정] group_id 우선 적용 (UI에서 넘어온 그룹번호 1~10)
        # =========================================================
        if group_id is None:
            group_id, group_label = self.classify_group_10(purchase_proba)
        else:
            # group_id 범위 방어
            group_id = int(group_id)
            group_id = min(max(group_id, 1), 10)

            if 1 <= group_id <= 5:
                group_label = f"그룹{group_id}(고위험 이탈군)"
            elif 6 <= group_id <= 8:
                group_label = f"그룹{group_id}(전환 기회군)"
            else:
                group_label = f"그룹{group_id}(구매 유력군)"
        # =========================================================

        # =========================================================
        # [기존 유지] 그룹별 메시지 10종
        # =========================================================
        messages = {
            # [고위험 이탈군: HIGH_RISK 1~5] - 어떻게든 붙잡아야 하는 절박함
            1: "🚨 [심폐소생술 시급] 고객님이 '뒤로 가기' 버튼과 썸 타는 중입니다! 혜택 한 줄 요약이랑 베스트 리뷰로 멱살 잡고 끌어와야 해요!",
            2: "🚪 '나 지금 나간다?'라고 온몸으로 외치는 중! 3초 안에 할인 쿠폰이나 무료배송 안 보여주면 영영 남남입니다. 빨리요!",
            3: "🧯 관심이라는 불씨가 생기기도 전에 로그아웃 각! 랜딩 페이지에 인기 상품이랑 신뢰 팍팍 가는 인증마크로 도배해서 눈길을 뺏으세요!",
            4: "🪝 살짝 솔깃해 보이지만, 로딩 1초만 늦어도 떠날 분입니다. 복잡한 거 다 빼고 핵심 혜택만 코앞에 들이미세요!",
            5: "⚠️ 이 정도면 '밀당' 고수네요. 살까 말까 고민하는 게 보입니다. '오늘만 이 가격' 콤보 한 방이면 바로 넘어옵니다!",

            # [전환 기회군: OPPORTUNITY 6~8] - 조금만 더 부추기면 사는 구간
            6: "👀 장바구니에 넣을까 말까 100번 고민 중! '최저가 보장'이나 '빠른 배송' 정보로 고객님의 우유부단함에 마침표를 찍어주세요!",
            7: "🎯 대어 낚기 직전입니다! '사람들이 이 제품 칭찬을 이렇게 많이 해요'라고 사회적 증거(후기/별점)를 마구 투척하세요!",
            8: "🔥 [결제 직전] 조금만 밀면 카드 슬래시! 한정판 쿠폰이나 '무료배송까지 얼마 안 남았어요'라는 멘트로 불을 지피세요!",

            # [구매 유력군: LIKELY_BUYER 9~10] - 방해하지 말고 결제 길만 깔아주기
            9: "🛒 이미 마음은 결제 완료! 괜히 팝업 띄워서 방해하지 말고, 쿠폰 자동 적용해서 레드카펫 깔아드립시다. 결제 길만 걷게 하세요!",
            10: "✅ [확정 전환] 이분은 숨만 쉬어도 구매하실 분입니다! 추가 영업은 사치일 뿐. 가볍게 '함께 사면 좋은 꿀템' 하나만 슥- 던져보세요."
        }

        base_msg = messages.get(group_id, "👤 현재는 관찰이 필요한 세션입니다.")

        # =========================================================
        # [기존 유지] 행동 기반 힌트
        # =========================================================
        hints = []
        if bounce > 0.5:
            hints.append("힌트: 이탈률(BounceRates)이 높아 ‘첫 화면 설득’이 최우선입니다.")
        if exit_r > 0.5:
            hints.append("힌트: 종료율(ExitRates)이 높아 ‘결제/최종 단계 마찰’ 점검이 필요합니다.")
        if page_val <= 0:
            hints.append("힌트: PageValues가 낮아 ‘가치/혜택 체감’이 부족할 가능성이 큽니다.")
        if duration > 60:
            hints.append("힌트: 체류시간이 길어 ‘관심은 있는데 마지막 확신이 부족’한 패턴일 수 있습니다.")

        if hints:
            return f"{group_label}\n{base_msg}\n\n" + "\n".join(f"- {h}" for h in hints)
        return f"{group_label}\n{base_msg}"
