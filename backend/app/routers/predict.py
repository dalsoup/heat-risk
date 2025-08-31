# backend/app/routers/predict.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional, Tuple, Dict
import os
import pandas as pd
import numpy as np

try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

router = APIRouter(tags=["predict"])

# ---------- 경로 해석 ----------
PROJECT_BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
PROJECT_ROOT = PROJECT_BACKEND_DIR.parent

# 환경변수로 강제 지정 가능
ENV_FEATURES_LATEST = os.getenv("FEATURES_INFER_LATEST", "")  # 절대/상대 경로 모두 허용

# 기본 탐색 후보 (오탈자까지 포함)
DEFAULT_BASES = [
    PROJECT_ROOT / "train" / "personal_features_infer_latest",
    PROJECT_ROOT / "train" / "personal_infer_features_latest",  # 오타 대응
    PROJECT_ROOT / "data"  / "train" / "personal_features_infer_latest",
]

# 모델 경로: 환경변수 우선, 없으면 backend/models/*_latest.joblib
ENV_LOGREG = os.getenv("MODEL_LOGREG_PATH", "")
ENV_XGB    = os.getenv("MODEL_XGB_PATH", "")
DEFAULT_MODEL_DIR = PROJECT_ROOT / "backend" / "models"
DEFAULT_LOGREG = DEFAULT_MODEL_DIR / "logreg_personal_latest.joblib"
DEFAULT_XGB    = DEFAULT_MODEL_DIR / "xgb_personal_latest.joblib"


# ---------- 파일/모델 로딩 ----------
def _resolve_features_latest() -> Path:
    if ENV_FEATURES_LATEST:
        p = Path(ENV_FEATURES_LATEST)
        if p.exists():
            return p
        raise HTTPException(status_code=503, detail=f"Features file not found: {p}")
    for base in DEFAULT_BASES:
        if base.suffix:
            if base.exists():
                return base
        else:
            for cand in (base.with_suffix(".csv"), base.with_suffix(".parquet")):
                if cand.exists():
                    return cand
    raise HTTPException(status_code=503, detail="No personal_features_infer_latest.* under /train (or alias)")

def _read_any(p: Path) -> pd.DataFrame:
    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(p)
        # 비표준 확장자 대응
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read features: {e}")

def _load_artifact(path: Path):
    if joblib_load is None:
        raise HTTPException(status_code=500, detail="joblib is not available to load model artifacts.")
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Model artifact missing: {path}")
    try:
        return joblib_load(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model artifact: {e}")

def _unpack_artifact(obj) -> Tuple[object, Optional[dict]]:
    """
    저장 포맷이 dict({'model','preproc',...}) 또는 모델 단일 객체 양쪽 지원.
    Returns: (model, preproc or None)
    """
    if isinstance(obj, dict):
        m = obj.get("model")
        pre = obj.get("preproc")
        if m is None or pre is None:
            # 일부 구버전 아티팩트: 모델만 있는 경우
            return obj.get("model", obj), obj.get("preproc", None)
        return m, pre
    return obj, None


def _load_models() -> Tuple[object, Optional[object], dict]:
    logreg_path = Path(ENV_LOGREG) if ENV_LOGREG else DEFAULT_LOGREG
    xgb_path    = Path(ENV_XGB)    if ENV_XGB    else DEFAULT_XGB

    lr_art = _load_artifact(logreg_path)
    LOGREG, PREPROC = _unpack_artifact(lr_art)
    if LOGREG is None or PREPROC is None:
        raise HTTPException(status_code=500, detail="logreg artifact must contain 'model' and 'preproc'")

    XGB = None
    if xgb_path.exists():
        try:
            x_art = _load_artifact(xgb_path)
            XGB, _ = _unpack_artifact(x_art)
        except Exception:
            XGB = None  # xgboost 불가 시 LR만 사용
    return LOGREG, XGB, PREPROC


# 전역 로드(앱 시작 시 1회)
LOGREG, XGB, PREPROC = _load_models()


# ---------- 전처리 ----------
def _apply_preprocess(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    cols_keep = preproc.get("cols_keep", [])
    medians   = preproc.get("medians", {})
    scaler    = preproc.get("scaler", None)
    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan
    X = df[cols_keep].copy().replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians.get(c, 0.0))
    return scaler.transform(X.values) if scaler is not None else X.values

def _apply_preprocess_noscale(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    cols_keep = preproc.get("cols_keep", [])
    medians   = preproc.get("medians", {})
    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan
    X = df[cols_keep].copy().replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians.get(c, 0.0))
    return X.values

def _predict_proba_safe(model, X_nd: np.ndarray) -> Optional[np.ndarray]:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_nd)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            if isinstance(proba, np.ndarray) and proba.ndim == 1:
                return proba
    except Exception:
        pass
    try:
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function(X_nd), dtype=float)
            return 1.0 / (1.0 + np.exp(-s))
    except Exception:
        pass
    try:
        pred = model.predict(X_nd)
        return np.asarray(pred, dtype=float)
    except Exception:
        pass
    return None


# ---------- 섹션 추출(DetailModal용) ----------
def _extract_sections(row: pd.Series, select: str = "meta,health,self_report") -> Tuple[Dict, Dict, Dict]:
    """
    한 유저(row)에서 meta/health/self_report 섹션을 best-effort로 추출.
    - 컬럼명 토큰 매칭 기반 버킷팅
    """
    if row is None or row.empty:
        return {}, {}, {}

    toks = [t.strip().lower() for t in (select or "").split(",") if t.strip()]
    want_meta   = any(t in ("meta",) for t in toks)
    want_health = any(t in ("health",) for t in toks)
    want_self   = any(t in ("self", "self_report", "report") for t in toks)

    meta, health, self_report = {}, {}, {}

    for k, v in row.to_dict().items():
        lk = str(k).lower()

        # 핵심 키/리스크 열 제외
        if lk in ("user_id", "adm_cd2", "adm_cd", "adm_nm", "dt_hour", "date", "time",
                  "risk_any", "risk_logreg", "risk_xgb", "risk_score", "risk_heat", "risk"):
            continue

        if want_meta and ("meta" in lk):
            meta[k] = v; continue

        if want_health and ("health" in lk or lk in {
            "hr_bpm", "stress_0_1", "symptom_score", "hr_base", "fitness", "vulnerability",
            "wbgt_c", "hi_c", "temp_c", "rh_pct", "hours_wbgt_ge28_last6h"
        }):
            health[k] = v; continue

        if want_self and ("self" in lk or "report" in lk or lk in {"reported", "last_dt"}):
            self_report[k] = v; continue

    # self_report 보정
    if "reported" not in {k.lower() for k in self_report.keys()} and "reported" in row:
        self_report["reported"] = row.get("reported")
    if "last_dt" not in {k.lower() for k in self_report.keys()} and "last_dt" in row:
        self_report["last_dt"] = row.get("last_dt")

    return meta, health, self_report


# ---------- API: /predict ----------
@router.get("/predict")
def get_predict(
    user_id: str = Query(..., description="user_id in personal_features_infer_latest"),
    select: str = Query("meta,health,self_report", description="섹션 힌트(쉼표)"),
):
    # 1) 최신 피처 로드
    feats_path = _resolve_features_latest()
    df = _read_any(feats_path)

    if "user_id" not in df.columns:
        raise HTTPException(status_code=500, detail="Features table missing 'user_id' column.")

    row_df = df[df["user_id"].astype(str) == str(user_id)].copy()
    if row_df.empty:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found.")

    # adm_cd2 문자열 보정
    if "adm_cd2" in row_df.columns:
        row_df["adm_cd2"] = row_df["adm_cd2"].astype(str)

    # 2) dt_hour 파싱(있을 때만)
    dt_val = None
    if "dt_hour" in row_df.columns:
        try:
            dts = pd.to_datetime(row_df["dt_hour"], errors="coerce")
            if pd.notna(dts.iloc[0]):
                dt_val = dts.iloc[0].strftime("%Y-%m-%dT%H:%M:%S")
            else:
                dt_val = str(row_df["dt_hour"].iloc[0])
        except Exception:
            dt_val = str(row_df["dt_hour"].iloc[0])

    # 3) 예측 (훈련 전처리 그대로 적용)
    X_sc  = _apply_preprocess(row_df.copy(), PREPROC)
    p_lr  = _predict_proba_safe(LOGREG, X_sc)
    risk_logreg = float(p_lr[0]) if p_lr is not None and len(p_lr) >= 1 else None

    risk_xgb: Optional[float] = None
    if XGB is not None:
        X_xgb = _apply_preprocess_noscale(row_df.copy(), PREPROC)
        p_xgb = _predict_proba_safe(XGB, X_xgb)
        if p_xgb is not None and len(p_xgb) >= 1:
            risk_xgb = float(p_xgb[0])

    # risk_any
    cand = [v for v in [risk_logreg, risk_xgb] if v is not None]
    risk_any = float(np.mean(cand)) if cand else None

    # 4) 섹션 생성
    row = row_df.iloc[0]
    meta, health, self_report = _extract_sections(row, select)

    # 5) 응답
    return {
        "user_id": str(user_id),
        "dt_hour": dt_val,
        "risk_logreg": risk_logreg,
        "risk_xgb": risk_xgb,
        "risk_any": risk_any,
        "meta": meta or None,
        "health": health or None,
        "self_report": self_report or None,
    }
