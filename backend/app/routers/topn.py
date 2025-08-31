# backend/app/routers/topn.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Literal
from pathlib import Path
import os
import pandas as pd
import numpy as np
from joblib import load

router = APIRouter(tags=["topn"])

PROJECT_BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PROJECT_BACKEND_DIR.parent

ENV_FEATURES_LATEST = os.getenv("FEATURES_INFER_LATEST", "")

DEFAULT_BASES = [
    PROJECT_ROOT / "train" / "personal_features_infer_latest",
    PROJECT_ROOT / "train" / "personal_infer_features_latest",
    PROJECT_ROOT / "data"  / "train" / "personal_features_infer_latest",
]

ENV_LOGREG = os.getenv("MODEL_LOGREG_PATH", "")
ENV_XGB    = os.getenv("MODEL_XGB_PATH", "")
DEFAULT_MODEL_DIR = PROJECT_ROOT / "backend" / "models"
DEFAULT_LOGREG = DEFAULT_MODEL_DIR / "logreg_personal_latest.joblib"
DEFAULT_XGB    = DEFAULT_MODEL_DIR / "xgb_personal_latest.joblib"

def _resolve_features_latest() -> Path:
    if ENV_FEATURES_LATEST:
        p = Path(ENV_FEATURES_LATEST)
        if p.exists(): return p
        raise HTTPException(status_code=503, detail=f"Features file not found: {p}")
    for base in DEFAULT_BASES:
        if base.suffix:
            if base.exists(): return base
        else:
            for cand in (base.with_suffix(".csv"), base.with_suffix(".parquet")):
                if cand.exists(): return cand
    raise HTTPException(status_code=503, detail="No personal_features_infer_latest.* under /train (or alias)")

def _read_any(p: Path) -> pd.DataFrame:
    try:
        if p.suffix.lower() == ".csv":    return pd.read_csv(p)
        if p.suffix.lower() in (".parquet", ".pq"): return pd.read_parquet(p)
        try:    return pd.read_parquet(p)
        except: return pd.read_csv(p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read features: {e}")

def _load_models():
    logreg_path = Path(ENV_LOGREG) if ENV_LOGREG else DEFAULT_LOGREG
    xgb_path    = Path(ENV_XGB)    if ENV_XGB    else DEFAULT_XGB
    if not logreg_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing model: {logreg_path}")
    try:
        lr_art = load(logreg_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load logreg: {e}")
    logreg = lr_art.get("model")
    preproc = lr_art.get("preproc")
    if logreg is None or preproc is None:
        raise HTTPException(status_code=500, detail="logreg artifact must contain 'model' and 'preproc'")

    xgb = None
    if xgb_path.exists():
        try:
            x_art = load(xgb_path)
            xgb = x_art.get("model")
        except Exception:
            xgb = None
    return logreg, xgb, preproc

LOGREG, XGB, PREPROC = _load_models()

def _apply_preprocess(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    cols_keep = preproc["cols_keep"]
    medians = preproc["medians"]
    scaler = preproc["scaler"]
    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan
    X = df[cols_keep].copy().replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians.get(c, 0.0))
    return scaler.transform(X.values)

def _apply_preprocess_noscale(df: pd.DataFrame, preproc: dict) -> np.ndarray:
    cols_keep = preproc["cols_keep"]
    medians = preproc["medians"]
    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan
    X = df[cols_keep].copy().replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians.get(c, 0.0))
    return X.values

@router.get("/topn")
def get_topn(
    n: int = Query(20, ge=1, le=1000),
    rank_by: Literal["risk_logreg", "risk_xgb"] = Query("risk_logreg"),
):
    feats_path = _resolve_features_latest()
    df = _read_any(feats_path)

    if df.empty:
        raise HTTPException(status_code=500, detail="Features table is empty.")
    if "user_id" not in df.columns:
        raise HTTPException(status_code=500, detail="Features table missing 'user_id' column.")

    # 시간 포맷(있으면 전달)
    if "dt_hour" in df.columns:
        try:
            dts = pd.to_datetime(df["dt_hour"], errors="coerce")
            df["dt_hour"] = dts.dt.strftime("%Y-%m-%dT%H:%M:%S").where(dts.notna(), None)
        except Exception:
            pass

    # 벡터 계산
    X_sc = _apply_preprocess(df.copy(), PREPROC)
    df["risk_logreg"] = LOGREG.predict_proba(X_sc)[:, 1]

    if XGB is not None:
        X_xgb = _apply_preprocess_noscale(df.copy(), PREPROC)
        df["risk_xgb"] = XGB.predict_proba(X_xgb)[:, 1]
    else:
        df["risk_xgb"] = np.nan

    # 정렬 & 상위 n
    rank_col = rank_by if rank_by in df.columns else "risk_logreg"
    df_sorted = df.sort_values(rank_col, ascending=False, na_position="last").head(n)

    items = []
    for _, row in df_sorted.iterrows():
        items.append(
            {
                "user_id": str(row.get("user_id")),
                "adm_cd2": (None if pd.isna(row.get("adm_cd2")) else str(row.get("adm_cd2"))),
                "dt_hour": (None if pd.isna(row.get("dt_hour")) else str(row.get("dt_hour"))),
                "risk_logreg": float(row.get("risk_logreg")),
                "risk_xgb": (None if pd.isna(row.get("risk_xgb")) else float(row.get("risk_xgb"))),
            }
        )
    return {"rank_by": rank_col, "n": n, "items": items}
