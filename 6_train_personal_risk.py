# train_personal_risk.py
# 개인 맞춤 온열질환 위험도 베이스라인 (LogisticRegression + optional XGBoost)

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from shutil import copyfile

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# --- Optional: XGBoost ---
HAS_XGB = True
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    HAS_XGB = False

# -------------------------
# Paths & config
# -------------------------
KST = timezone(timedelta(hours=9))

# 프로젝트 루트(heat-risk/) 기준 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[0]  # 파일이 루트에 있다고 가정
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
PRED_DIR = DATA_DIR / "predictions"

# 모델 산출물은 backend/models 로 저장 (환경변수 MODEL_OUTPUT_DIR로 오버라이드 가능)
MODEL_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", PROJECT_ROOT / "backend" / "models"))
for d in [PRED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

F_TRAIN = Path(os.getenv("TRAIN_TABLE", TRAIN_DIR / "personal_features_train_latest.csv"))
F_INFER = Path(os.getenv("INFER_TABLE", TRAIN_DIR / "personal_features_infer_latest.csv"))

TS = datetime.now(KST).strftime("%Y%m%d%H%M%S")

# Feature sets
WEAR_SOFT = ["hr_bpm", "stress_0_1", "wearing", "reported", "symptom_score"]
WEATHER   = ["wbgt_c", "hi_c", "hours_wbgt_ge28_last6h", "temp_c", "rh_pct"]
PROFILE   = ["adherence", "hr_base", "fitness", "vulnerability"]
TIME_FEAT = ["hour", "dow"]

USE_COLS   = WEAR_SOFT + WEATHER + PROFILE + TIME_FEAT
TARGET_COL = "hard_label"

# Options
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.2"))       # 20%
RANDOM_SEED = int(os.getenv("SEED", "2025"))
SCALE_POS_WEIGHT = os.getenv("SCALE_POS_WEIGHT", "") # ""이면 자동 계산
TOPN = int(os.getenv("INFER_TOPN", "20"))            # 추론 상위 N 출력

# -------------------------
# Utils
# -------------------------
def _require(path: Path, name: str):
    if not path.exists():
        raise SystemExit(f"Missing {name}: {path}")

def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def _select_existing(df: pd.DataFrame, cols: list):
    return [c for c in cols if c in df.columns]

def _safe_float(df: pd.DataFrame, cols: list):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _report_metrics(y_true, y_prob, prefix="VAL"):
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    ap  = average_precision_score(y_true, y_prob)
    print(f"[{prefix}] ROC-AUC={auc:.4f}  AP={ap:.4f}")
    print(f"[{prefix}] Report:\n", classification_report(y_true, y_pred, digits=3))

def _nan_guard(name, arr_like):
    if isinstance(arr_like, pd.DataFrame):
        if not np.isfinite(arr_like.values).all():
            bad_cols = [c for c in arr_like.columns if not np.isfinite(arr_like[c].values).all()]
            raise ValueError(f"{name}: non-finite values remain in columns {bad_cols}")
    else:
        if not np.isfinite(arr_like).all():
            raise ValueError(f"{name}: non-finite values remain.")

# -------------------------
# Robust preprocessing
# -------------------------
def safe_fit_preprocess(df: pd.DataFrame, feature_cols: list):
    """
    훈련 데이터로 전처리 사전(fit):
      - Inf → NaN
      - 중앙값 대치(열 전체 NaN이면 0 사용)
      - 0-분산 컬럼 제거
      - 표준화 스케일러 fit
    Returns:
      preproc: dict with keys
        - cols_keep: list[str] (0-분산 제거 후 최종 사용 열)
        - medians: dict[col -> float] (중앙값, 전체 NaN이면 0)
        - scaler: StandardScaler
    """
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # 중앙값 사전 계산
    medians = {}
    for c in X.columns:
        med = np.nanmedian(X[c].values)
        if np.isnan(med):
            med = 0.0
        medians[c] = float(med)
        X[c] = X[c].fillna(med)

    # 0-분산 컬럼 제거
    stds = X.std(axis=0, ddof=0)
    cols_keep = [c for c in X.columns if np.isfinite(stds[c]) and stds[c] > 0.0]
    if len(cols_keep) == 0:
        raise SystemExit("All features have zero variance after imputation. Check your data.")

    Xk = X[cols_keep].copy()
    _nan_guard("TRAIN_X_after_impute", Xk)

    scaler = StandardScaler()
    Xk_sc = scaler.fit_transform(Xk)
    _nan_guard("TRAIN_X_after_scaler", Xk_sc)

    return {
        "cols_keep": cols_keep,
        "medians": medians,
        "scaler": scaler,
    }

def apply_preprocess(df: pd.DataFrame, preproc: dict):
    """
    검증/추론 데이터에 훈련 전처리 적용:
      - 동일 컬럼셋
      - 동일 중앙값 대치
      - 동일 스케일러 transform
    """
    cols_keep = preproc["cols_keep"]
    medians   = preproc["medians"]
    scaler    = preproc["scaler"]

    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan

    X = df[cols_keep].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(medians.get(c, 0.0))

    _nan_guard("APPLY_X_after_impute", X)
    X_sc = scaler.transform(X.values)
    _nan_guard("APPLY_X_after_scaler", X_sc)
    return X_sc

def apply_preprocess_noscale(df: pd.DataFrame, preproc: dict):
    """
    XGBoost 입력용(스케일 없이 동일한 대치/컬럼 선택만 적용)
    """
    cols_keep = preproc["cols_keep"]
    medians   = preproc["medians"]
    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan
    X = df[cols_keep].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(medians.get(c, 0.0))
    _nan_guard("APPLY_XGB_X_after_impute", X)
    return X.values

# -------------------------
# 1) Load
# -------------------------
def _load_tables():
    _require(F_TRAIN, "train table")
    train = pd.read_csv(F_TRAIN)
    if TARGET_COL not in train.columns:
        raise SystemExit(f"Train table must contain '{TARGET_COL}'.")
    if "dt_hour" in train.columns:
        train["dt_hour"] = _to_dt(train["dt_hour"])
        train = train.dropna(subset=["dt_hour", TARGET_COL]).copy()
    else:
        train = train.dropna(subset=[TARGET_COL]).copy()

    num_cols = _select_existing(train, USE_COLS)
    train = _safe_float(train, num_cols)

    _require(F_INFER, "infer table")
    infer = pd.read_csv(F_INFER)
    if "dt_hour" in infer.columns:
        infer["dt_hour"] = _to_dt(infer["dt_hour"])
    infer = _safe_float(infer, _select_existing(infer, USE_COLS))
    return train, infer, num_cols

train, infer, num_cols = _load_tables()

# -------------------------
# 2) Train/Val split (time-based holdout)
# -------------------------
if "dt_hour" in train.columns:
    train_sorted = train.sort_values("dt_hour")
else:
    train_sorted = train.copy()

cut_idx = int(len(train_sorted) * (1.0 - VAL_SIZE))
train_df = train_sorted.iloc[:cut_idx].copy()
val_df   = train_sorted.iloc[cut_idx:].copy()

y_tr = train_df[TARGET_COL].astype(int).values
y_val = val_df[TARGET_COL].astype(int).values

# -------------------------
# 3) Fit preprocess, apply
# -------------------------
preproc = safe_fit_preprocess(train_df, _select_existing(train_df, num_cols))
X_tr_sc = apply_preprocess(train_df, preproc)
X_val_sc = apply_preprocess(val_df, preproc)

# class weight
if SCALE_POS_WEIGHT.strip() == "":
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(k): float(v) for k, v in zip(classes, cw)}
    print("[INFO] class_weight (balanced):", class_weight)
else:
    sw = float(SCALE_POS_WEIGHT)
    class_weight = {0: 1.0, 1: sw}
    print(f"[INFO] class_weight from SCALE_POS_WEIGHT={sw}")

# -------------------------
# 4) Train models
# -------------------------
# (A) Logistic Regression
logreg = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    class_weight=class_weight,
    random_state=RANDOM_SEED,
)
logreg.fit(X_tr_sc, y_tr)
p_val_lr = logreg.predict_proba(X_val_sc)[:, 1]
_report_metrics(y_val, p_val_lr, prefix="VAL-LogReg")

# Save LR artifacts (딕셔너리로 저장: router가 그대로 읽어서 사용)
dump({"model": logreg, "preproc": preproc, "features_all": num_cols},
     MODEL_DIR / f"logreg_personal_{TS}.joblib")
dump({"model": logreg, "preproc": preproc, "features_all": num_cols},
     MODEL_DIR / "logreg_personal_latest.joblib")

# (B) XGBoost (optional)
if HAS_XGB:
    pos_weight = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
    if SCALE_POS_WEIGHT.strip() != "":
        pos_weight = float(SCALE_POS_WEIGHT)

    X_tr_xgb = apply_preprocess_noscale(train_df, preproc)
    X_val_xgb = apply_preprocess_noscale(val_df, preproc)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        learning_rate=0.05,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=RANDOM_SEED,
        scale_pos_weight=pos_weight,
        n_jobs=0,
    )
    xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
    p_val_xgb = xgb.predict_proba(X_val_xgb)[:, 1]
    _report_metrics(y_val, p_val_xgb, prefix="VAL-XGB")

    dump({"model": xgb, "preproc": preproc, "features_all": num_cols},
         MODEL_DIR / f"xgb_personal_{TS}.joblib")
    dump({"model": xgb, "preproc": preproc, "features_all": num_cols},
         MODEL_DIR / "xgb_personal_latest.joblib")
else:
    print("[WARN] xgboost not installed; skipping XGB model.")

# -------------------------
# 5) Save validation predictions  (latest = 실제 CSV 복사본)
# -------------------------
if "dt_hour" in val_df.columns:
    val_out = val_df[["user_id", "adm_cd2", "dt_hour", TARGET_COL]].copy()
else:
    val_out = val_df[["user_id", "adm_cd2", TARGET_COL]].copy()
    val_out["dt_hour"] = pd.NaT

val_out["p_logreg"] = p_val_lr
if HAS_XGB:
    val_out["p_xgb"] = p_val_xgb

val_path_ts = PRED_DIR / f"val_predictions_{TS}.csv"
val_latest  = PRED_DIR / "val_predictions_latest.csv"

val_out.to_csv(val_path_ts, index=False)
copyfile(val_path_ts, val_latest)
print(f"[SAVE] {val_path_ts.name}  (also wrote -> {val_latest.name})")

# -------------------------
# 6) Inference on 'now' table  (latest = 실제 CSV 복사본)
# -------------------------
# NOTE: 라우터에서 preproc을 그대로 쓰기 때문에 여기서 굳이 risk_any가 없어도 되지만,
#       디버깅/직접 확인 편의를 위해 infer CSV에도 risk_any를 같이 저장합니다.
X_inf_sc = apply_preprocess(infer, preproc)
infer["risk_logreg"] = logreg.predict_proba(X_inf_sc)[:, 1]

if HAS_XGB:
    X_inf_xgb = apply_preprocess_noscale(infer, preproc)
    infer["risk_xgb"] = xgb.predict_proba(X_inf_xgb)[:, 1]

# risk_any = 사용 가능한 위험도 평균
cand_cols = [c for c in ["risk_logreg", "risk_xgb", "risk_score", "risk_heat", "risk"] if c in infer.columns]
if cand_cols:
    infer["risk_any"] = infer[cand_cols].mean(axis=1, skipna=True)
else:
    infer["risk_any"] = np.nan

inf_path_ts = PRED_DIR / f"infer_predictions_{TS}.csv"
inf_latest  = PRED_DIR / "infer_predictions_latest.csv"

infer.to_csv(inf_path_ts, index=False)
copyfile(inf_path_ts, inf_latest)
print(f"[SAVE] {inf_path_ts.name}  (also wrote -> {inf_latest.name})")

# -------------------------
# 7) Show top-N at now
# -------------------------
rank_col = "risk_xgb" if (HAS_XGB and "risk_xgb" in infer.columns) else "risk_logreg"
cols_show = ["user_id", "adm_cd2", "dt_hour", rank_col] if "dt_hour" in infer.columns else ["user_id", "adm_cd2", rank_col]
topn = infer.sort_values(rank_col, ascending=False).head(TOPN)[cols_show]
print(f"\n[TOP {TOPN}] by {rank_col}")
print(topn.to_string(index=False))
