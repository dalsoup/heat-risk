# backend/app/routers/health.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field, conint

router = APIRouter(prefix="/health", tags=["health"])

# ----------------------------------------------------------------------
# Timezone
# ----------------------------------------------------------------------
KST = timezone(timedelta(hours=9))

def now_kst_iso() -> str:
    return datetime.now(KST).isoformat()

# ----------------------------------------------------------------------
# In-memory demo state (프론트 개발용 임시 스토리지)
# 실제 서비스에서는 DB/캐시로 대체
# ----------------------------------------------------------------------
_HYDRATION_STATE: Dict[str, int] = {
    "current_ml": 1000,  # 기본값
}
_HYDRATION_GOAL = 2000

# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class Location(BaseModel):
    adm_cd2: str = Field(..., example="1111051500")
    dong_name: str = Field(..., example="청운효자동")

class Heat(BaseModel):
    headline: str = Field(..., example="폭염")
    risk_score: conint(ge=0, le=100) = 40
    score_breakdown: Dict[str, float] = Field(
        default_factory=lambda: {"water": 50.0, "cool_rest": 10.0, "ambient_c": 40.0}
    )

class Hydration(BaseModel):
    current_ml: int = Field(..., example=1000)
    goal_ml: int = Field(..., example=2000)

class Shelters(BaseModel):
    nearby_count: int = Field(..., example=2)
    radius_m: int = Field(..., example=500)

class Vitals(BaseModel):
    hr_bpm: Optional[int] = Field(None, example=99)
    hr_range: List[int] = Field(default_factory=lambda: [70, 100])
    spo2_pct: Optional[int] = Field(None, example=95)
    spo2_range: List[int] = Field(default_factory=lambda: [90, 100])

class Sleep(BaseModel):
    last_night_hours: Optional[float] = Field(9, example=9)

class HealthSummary(BaseModel):
    location: Location
    heat: Heat
    hydration: Hydration
    shelters: Shelters
    vitals: Vitals
    sleep: Sleep
    updated_at: str

class HydrationAddReq(BaseModel):
    amount_ml: conint(ge=1, le=2000) = Field(250, description="추가 섭취량(ml)")

class HydrationAddResp(BaseModel):
    current_ml: int
    goal_ml: int
    updated_at: str

class SelfReportReq(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    vitals: Optional[Dict[str, float]] = None
    context: Optional[Dict[str, float]] = None

class SelfReportResp(BaseModel):
    level: str  # "mild" | "severe"
    headline: str
    recommendations: List[str]
    actions: Dict[str, bool]

class AlertResp(BaseModel):
    has_alert: bool
    title: str
    cta: List[str] = Field(default_factory=list)

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@router.get("/", summary="헬스 라우터 확인용")
def ok():
    return {"ok": True}

@router.get("/summary", response_model=HealthSummary, summary="홈 화면용 요약")
def get_summary():
    """
    프론트 홈 카드에 필요한 요약 정보를 한번에 반환.
    수분 섭취량은 인메모리 상태(_HYDRATION_STATE)를 사용.
    """
    return HealthSummary(
        location=Location(adm_cd2="1111051500", dong_name="청운효자동"),
        heat=Heat(
            headline="폭염",
            risk_score=40,
            score_breakdown={"water": 50.0, "cool_rest": 10.0, "ambient_c": 40.0},
        ),
        hydration=Hydration(
            current_ml=int(_HYDRATION_STATE.get("current_ml", 1000)),
            goal_ml=_HYDRATION_GOAL,
        ),
        shelters=Shelters(nearby_count=2, radius_m=500),
        vitals=Vitals(hr_bpm=99, hr_range=[70, 100], spo2_pct=95, spo2_range=[90, 100]),
        sleep=Sleep(last_night_hours=9),
        updated_at=now_kst_iso(),
    )

@router.get("/alert", response_model=AlertResp, summary="실시간 경고(폴링용)")
def get_alert():
    """
    개발용 기본값: 경고 없음.
    테스트하고 싶으면 has_alert=True 로 변경 후 UI 확인.
    """
    return AlertResp(
        has_alert=False,
        title="온열질환 주의!",
        cta=["self_check", "find_shelter"],
    )

@router.post("/hydration/add", response_model=HydrationAddResp, summary="수분 섭취 기록")
def add_hydration(payload: HydrationAddReq = Body(...)):
    """
    인메모리 섭취량을 증가시키고 최신 상태를 반환.
    실제 서비스에서는 사용자별/일자별로 DB에 적재하세요.
    """
    _HYDRATION_STATE["current_ml"] = int(_HYDRATION_STATE.get("current_ml", 0)) + int(payload.amount_ml)
    # 과도한 누적 방지: 목표의 300%에서 캡 (임시)
    cap = _HYDRATION_GOAL * 3
    if _HYDRATION_STATE["current_ml"] > cap:
        _HYDRATION_STATE["current_ml"] = cap

    return HydrationAddResp(
        current_ml=_HYDRATION_STATE["current_ml"],
        goal_ml=_HYDRATION_GOAL,
        updated_at=now_kst_iso(),
    )

@router.post("/self-report", response_model=SelfReportResp, summary="자가진단 제출")
def self_report(payload: SelfReportReq):
    """
    간단한 규칙 기반 분류:
      - 증상에 ["의식 저하","구토","극심한 두통"] 중 하나라도 포함 → severe
      - 아니면 mild
    """
    syms = set(s.strip() for s in payload.symptoms or [])
    severe = any(s in syms for s in ["의식 저하", "구토", "극심한 두통"])

    if severe:
        return SelfReportResp(
            level="severe",
            headline="심각한 온열질환",
            recommendations=[
                "모든 활동을 즉시 중단하고 서늘한 장소로 이동",
                "옷을 느슨하게 풀고 젖은 수건으로 몸 식히기",
                "주변인 도움 요청, 응급 증상 시 119 연락",
            ],
            actions={"shelter": True, "hospital": True, "call119": True},
        )

    return SelfReportResp(
        level="mild",
        headline="경미한 온열질환",
        recommendations=[
            "시원한 장소에서 10~15분 휴식",
            "시원한 물을 조금씩 섭취 (150~250ml)",
            "모자·양산으로 햇볕 노출 최소화",
        ],
        actions={"shelter": True, "hospital": True, "call119": False},
    )
