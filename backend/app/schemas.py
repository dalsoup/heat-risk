from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    infer_user_id: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    email: EmailStr
    infer_user_id: Optional[str]
    class Config:
        from_attributes = True

# === Features (serve_api와 동일 스키마) ===
class UserFeatures(BaseModel):
    # WEAR_SOFT
    hr_bpm: Optional[float] = None
    stress_0_1: Optional[float] = None
    wearing: Optional[int] = None
    reported: Optional[int] = None
    symptom_score: Optional[float] = None
    # WEATHER
    wbgt_c: float
    hi_c: float
    hours_wbgt_ge28_last6h: Optional[float] = 0.0
    temp_c: float
    rh_pct: float
    # PROFILE
    adherence: Optional[float] = 0.7
    hr_base: Optional[float] = 70.0
    fitness: Optional[float] = 0.0
    vulnerability: Optional[float] = 0.0
    # TIME_FEAT
    hour: int = Field(..., ge=0, le=23)
    dow: int = Field(..., ge=0, le=6)

class RiskResponse(BaseModel):
    risk_logreg: float
    risk_xgb: Optional[float] = None
