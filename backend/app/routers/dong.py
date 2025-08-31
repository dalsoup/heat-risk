# backend/app/routers/dong.py
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import ORJSONResponse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

router = APIRouter(prefix="/dong", tags=["dong"])

# ---------------------------
# Paths
# ---------------------------
BACKEND_DIR = Path(__file__).resolve().parents[2]   # .../backend
ROOT_DIR    = BACKEND_DIR.parent                    # project root

DATA_DIR      = ROOT_DIR / "data"
TRAIN_DIR     = DATA_DIR / "train"
FEATURES_BASE = TRAIN_DIR / "personal_features_infer_latest"  # .csv | .parquet

# 모델은 backend/models
MODELS_DIR  = BACKEND_DIR / "models"
LOGREG_PATH = MODELS_DIR / "logreg_personal_latest.joblib"
XGB_PATH    = MODELS_DIR / "xgb_personal_latest.joblib"

# ---------------------------
# Small cache
# ---------------------------
_MODEL_CACHE: Dict[str, dict] = {}
_CACHE_TTL_SEC = 60

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0

# ---------------------------
# IO utils
# ---------------------------
def _read_table_maybe(base: Path) -> pd.DataFrame:
    """base가 확장자 없으면 .csv → .parquet 순으로 시도."""
    try:
        if base.suffix:
            return _read_by_suffix(base)
        for cand in (base.with_suffix(".csv"), base.with_suffix(".parquet")):
            if cand.exists():
                return _read_by_suffix(cand)
    except Exception:
        pass
    return pd.DataFrame()

def _read_by_suffix(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.DataFrame()

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------------------------
# Artifact loading (dict-aware)
# ---------------------------
def _load_artifact(path: Path, key: str):
    """joblib에서 저장된 dict({'model','preproc',...}) 또는 모델 객체를 로드."""
    if joblib_load is None or not path.exists():
        return None
    now = time.time()
    entry = _MODEL_CACHE.get(key)
    mtime = _mtime(path)
    if entry and (now - entry["ts"] < _CACHE_TTL_SEC) and entry["mtime"] == mtime:
        return entry["obj"]
    obj = joblib_load(path)
    _MODEL_CACHE[key] = {"obj": obj, "ts": now, "mtime": mtime}
    return obj

# ---------------------------
# Preprocess (use training-time preproc)
# ---------------------------
def _apply_from_preproc_lr(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    """LR 입력: cols_keep + medians + StandardScaler.transform."""
    cols_keep = preproc.get("cols_keep", [])
    medians   = preproc.get("medians", {})
    scaler    = preproc.get("scaler", None)

    X = df.copy()
    for c in cols_keep:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols_keep].replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(medians.get(c, 0.0))

    if scaler is None:
        return X.values
    return scaler.transform(X.values)

def _apply_from_preproc_xgb(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    """XGB 입력: cols_keep + medians (스케일 없음)."""
    cols_keep = preproc.get("cols_keep", [])
    medians   = preproc.get("medians", {})
    X = df.copy()
    for c in cols_keep:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols_keep].replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(medians.get(c, 0.0))
    return X.values

def _predict_proba_safe(model, X_nd: np.ndarray) -> Optional[np.ndarray]:
    """predict_proba[:,1] → ndarray; 실패 시 None."""
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

# ---------------------------
# Presentation helpers
# ---------------------------
_ID_COLS = {"user_id","adm_cd2","adm_cd","adm_nm","dt","dt_hour","date","time"}

def _select_fields(df: pd.DataFrame, select: str) -> pd.DataFrame:
    """토큰 기반 열 간소화. 위험도/핵심키는 항상 유지."""
    if df.empty or not select:
        return df
    tokens = [t.strip().lower() for t in select.split(",") if t.strip()]
    if not tokens:
        return df
    keep: List[str] = []
    for c in df.columns:
        lc = c.lower()
        if lc in ("adm_cd2","adm_cd","adm_nm","dt","dt_hour","date","time","user_id"):
            keep.append(c); continue
        if lc.startswith("risk") or "risk" in lc or lc in ("risk_any","risk_logreg","risk_xgb","risk_score","risk_heat","risk"):
            keep.append(c); continue
        if any(tok in lc for tok in tokens):
            keep.append(c)
    for must in ("adm_cd2","adm_cd","adm_nm"):
        if must in df.columns and must not in keep:
            keep.append(must)
    keep = [c for c in keep if c in df.columns]
    return df if not keep else df[keep]

def _sort_order(order: str) -> bool:
    return (order or "desc").lower() != "asc"

def _safe_topk_users(group: pd.DataFrame, sort_by: str, k: int, descending: bool) -> List[Dict]:
    df = group.copy()
    by = sort_by if sort_by in df.columns else None
    if by is None:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        by = numeric_cols[0] if numeric_cols else None
    if by is not None:
        df = df.sort_values(by=by, ascending=not descending)
    base_cols = [c for c in ("user_id","dt_hour","risk_any","risk_logreg","risk_xgb","risk_score") if c in df.columns]
    if not base_cols:
        base_cols = df.columns.tolist()[:6]
    if "user_id" not in base_cols:
        base_cols = ["user_id"] + base_cols
    return df[base_cols].head(max(0, int(k))).to_dict(orient="records")

def _extract_sections(row: pd.Series, select: str = "meta,health,self_report"):
    """
    한 유저(row)에서 meta/health/self_report 섹션을 best-effort로 추출.
    - 컬럼명이 포함하는 토큰으로 버킷팅 (예: 'meta', 'health', 'self', 'report')
    - 못 찾으면 빈 dict 반환
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

        # 핵심 필드는 제외
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

    # self_report 형태 보정
    if "reported" not in {k.lower() for k in self_report.keys()} and "reported" in row:
        self_report["reported"] = row.get("reported")
    if "last_dt" not in {k.lower() for k in self_report.keys()} and "last_dt" in row:
        self_report["last_dt"] = row.get("last_dt")

    return meta, health, self_report

# ---------------------------
# Endpoint: /dong/all
# ---------------------------
@router.get("/all", response_class=ORJSONResponse)
def get_all(
    include_users: bool = Query(False, description="동별 상위 사용자 리스트 포함 여부"),
    max_users_per_dong: int = Query(500, ge=0, le=10000),
    user_sort_by: str = Query("risk_any", description="사용자 정렬 기준 컬럼명"),
    user_order: str = Query("desc", description="asc | desc"),
    select: str = Query("meta,health,self_report", description="포함할 컬럼군 힌트(쉼표로 분리)"),
):
    """
    personal_features_infer_latest + backend/models 최신 아티팩트(dict)를 로드해 즉시 추론,
    동별 요약(summary.mean.{risk_any, risk_logreg, risk_xgb})과 선택적 상위 사용자 목록을 반환.
    """
    try:
        # 1) 피처 로드
        feats = _read_table_maybe(FEATURES_BASE)
        feats = _normalize_columns(feats)
        if feats.empty:
            return {"rows": [], "meta": {"note": f"features not found: {str(FEATURES_BASE)}.*"}}

        # ID/키 보정
        if "user_id" not in feats.columns:
            feats = feats.copy()
            feats["user_id"] = [f"u{i:06d}" for i in range(len(feats))]
        key_col = "adm_cd2" if "adm_cd2" in feats.columns else ("adm_cd" if "adm_cd" in feats.columns else None)

        # adm_cd2 문자열 보장(프론트/GeoJSON 매칭 안정화)
        if "adm_cd2" in feats.columns:
            feats["adm_cd2"] = feats["adm_cd2"].astype(str)

        # 2) 모델 로드(dict-aware)
        lr_art  = _load_artifact(LOGREG_PATH, "logreg")
        xgb_art = _load_artifact(XGB_PATH, "xgb")

        if (lr_art is None) and (xgb_art is None):
            return {"rows": [], "meta": {"note": f"no model artifacts under {str(MODELS_DIR)}"}}

        feats_pred = feats.copy()

        # 3) 추론: LR
        if lr_art is not None:
            lr_model  = lr_art["model"] if isinstance(lr_art, dict) and "model" in lr_art else lr_art
            lr_pre    = lr_art.get("preproc") if isinstance(lr_art, dict) else None
            if lr_pre is not None:
                X_lr = _apply_from_preproc_lr(feats, lr_pre)
                p_lr = _predict_proba_safe(lr_model, X_lr)
                if p_lr is not None:
                    feats_pred["risk_logreg"] = p_lr

        # 4) 추론: XGB
        if xgb_art is not None:
            xgb_model = xgb_art["model"] if isinstance(xgb_art, dict) and "model" in xgb_art else xgb_art
            xgb_pre   = xgb_art.get("preproc") if isinstance(xgb_art, dict) else None
            if xgb_pre is not None:
                X_xgb = _apply_from_preproc_xgb(feats, xgb_pre)
                p_xgb = _predict_proba_safe(xgb_model, X_xgb)
                if p_xgb is not None:
                    feats_pred["risk_xgb"] = p_xgb

        # 5) risk_any = 사용 가능 위험도의 평균
        cand_cols = [c for c in ["risk_logreg","risk_xgb","risk_score","risk_heat","risk"] if c in feats_pred.columns]
        feats_pred["risk_any"] = feats_pred[cand_cols].mean(axis=1, skipna=True) if cand_cols else np.nan
        if feats_pred["risk_any"].isna().all() and "risk_logreg" in feats_pred.columns:
            feats_pred["risk_any"] = feats_pred["risk_logreg"]

        # 6) select 적용(위험도/핵심키는 항상 유지)
        view_df = _select_fields(feats_pred, select) if select else feats_pred

        # 7) 그룹핑
        if key_col:
            groups_keys = view_df[[key_col]].drop_duplicates()
        else:
            groups_keys = pd.DataFrame([{}])

        rows: List[Dict] = []
        descending = _sort_order(user_order)

        for _, krow in groups_keys.iterrows():
            item: Dict = {}
            if key_col:
                key_val = krow[key_col]
                item[key_col] = str(key_val)

                mask_full = feats_pred[key_col] == key_val
                mask_view = view_df[key_col] == key_val
                g_full = feats_pred.loc[mask_full]
                g_view = view_df.loc[mask_view]
            else:
                item["group"] = "all"
                g_full = feats_pred
                g_view = view_df

            # 대표 adm_nm
            if "adm_nm" in g_view.columns:
                try:
                    item["adm_nm"] = g_view["adm_nm"].mode(dropna=True).iloc[0]
                except Exception:
                    s = g_view["adm_nm"].dropna()
                    item["adm_nm"] = s.iloc[0] if not s.empty else None

            # summary.mean 핵심 위험도
            mean_any = float(g_full["risk_any"].mean())  if "risk_any" in g_full.columns  else None
            mean_lr  = float(g_full["risk_logreg"].mean()) if "risk_logreg" in g_full.columns else None
            mean_xgb = float(g_full["risk_xgb"].mean()) if "risk_xgb" in g_full.columns else None
            if (mean_any is None) or (pd.isna(mean_any)):
                pair = [v for v in [mean_lr, mean_xgb] if v is not None and not pd.isna(v)]
                mean_any = float(pd.Series(pair).mean()) if pair else None

            summary = {"mean": {}, "max": {}}
            if mean_any is not None and not pd.isna(mean_any):
                summary["mean"]["risk_any"] = mean_any
            if mean_lr is not None and not pd.isna(mean_lr):
                summary["mean"]["risk_logreg"] = mean_lr
            if mean_xgb is not None and not pd.isna(mean_xgb):
                summary["mean"]["risk_xgb"] = mean_xgb

            # 참고용: 전체 수치형 평균/최대 (관찰용)
            num_cols = g_full.select_dtypes("number").columns.tolist()
            if num_cols:
                summary["mean"].update(g_full[num_cols].mean(numeric_only=True).to_dict())
                summary["max"] = g_full[num_cols].max(numeric_only=True).to_dict()

            item["summary"] = summary

            # 상위 사용자
            if include_users:
                item["users"] = _safe_topk_users(
                    group=g_view,
                    sort_by=user_sort_by,
                    k=max_users_per_dong,
                    descending=descending,
                )

            rows.append(item)

        return {
            "rows": rows,
            "meta": {
                "features_path": str(FEATURES_BASE),
                "models_dir": str(MODELS_DIR),
                "has_logreg": bool(LOGREG_PATH.exists()),
                "has_xgb": bool(XGB_PATH.exists()),
                "include_users": include_users,
                "user_sort_by": user_sort_by,
                "user_order": user_order,
                "select": select,
                "groups": len(rows),
            },
        }

    except Exception as e:
        return {"rows": [], "error": str(e)}

# ---------------------------
# Endpoint: /dong/predict  (DetailModal에서 사용)
# ---------------------------
@router.get("/predict", response_class=ORJSONResponse)
def predict_user(
    user_id: str,
    select: str = Query("meta,health,self_report", description="섹션 힌트(쉼표)"),
):
    """
    단일 유저 위험도와 부가 정보 반환.
    - features: data/train/personal_features_infer_latest
    - models: backend/models/{logreg,xgb}_personal_latest.joblib  (dict: {'model','preproc',...})
    - 전처리: 학습 시 저장한 preproc 그대로 적용
    응답 예:
    {
      "user_id": "...",
      "dt_hour": "...",
      "risk_logreg": 0.44,
      "risk_xgb": 0.50,
      "risk_any": 0.47,
      "meta": {...},
      "health": {...},
      "self_report": {...}
    }
    """
    try:
        feats = _read_table_maybe(FEATURES_BASE)
        feats = _normalize_columns(feats)
        if feats.empty:
            return {"error": f"features not found: {str(FEATURES_BASE)}.*", "user_id": user_id}

        # user_id 존재 확인
        if "user_id" not in feats.columns:
            return {"error": "features table has no 'user_id' column", "user_id": user_id}

        dfu = feats.loc[feats["user_id"].astype(str) == str(user_id)].copy()
        if dfu.empty:
            return {"error": "user_id not found", "user_id": user_id}

        # adm_cd2 문자열 보정(일관성)
        if "adm_cd2" in dfu.columns:
            dfu["adm_cd2"] = dfu["adm_cd2"].astype(str)

        # 모델 로드
        lr_art  = _load_artifact(LOGREG_PATH, "logreg")
        xgb_art = _load_artifact(XGB_PATH, "xgb")

        risk_lr = None
        risk_xgb = None

        # LR 예측
        if lr_art is not None:
            lr_model = lr_art["model"] if isinstance(lr_art, dict) and "model" in lr_art else lr_art
            lr_pre   = lr_art.get("preproc") if isinstance(lr_art, dict) else None
            if lr_pre is not None:
                X_lr = _apply_from_preproc_lr(dfu, lr_pre)
                p_lr = _predict_proba_safe(lr_model, X_lr)
                if p_lr is not None and len(p_lr) >= 1:
                    risk_lr = float(p_lr[0])

        # XGB 예측
        if xgb_art is not None:
            xgb_model = xgb_art["model"] if isinstance(xgb_art, dict) and "model" in xgb_art else xgb_art
            xgb_pre   = xgb_art.get("preproc") if isinstance(xgb_art, dict) else None
            if xgb_pre is not None:
                X_xgb = _apply_from_preproc_xgb(dfu, xgb_pre)
                p_xgb = _predict_proba_safe(xgb_model, X_xgb)
                if p_xgb is not None and len(p_xgb) >= 1:
                    risk_xgb = float(p_xgb[0])

        # risk_any
        cand = [v for v in [risk_lr, risk_xgb] if v is not None]
        risk_any = float(np.mean(cand)) if cand else None

        # 섹션 만들기
        row = dfu.iloc[0]
        meta, health, self_report = _extract_sections(row, select)

        # 응답
        out = {
            "user_id": str(row.get("user_id")),
            "dt_hour": str(row.get("dt_hour")) if "dt_hour" in dfu.columns else None,
            "risk_logreg": risk_lr if risk_lr is not None else None,
            "risk_xgb": risk_xgb if risk_xgb is not None else None,
            "risk_any": risk_any,
            "meta": meta or None,
            "health": health or None,
            "self_report": self_report or None,
        }
        return out

    except Exception as e:
        return {"error": str(e), "user_id": user_id}
