import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from shutil import copyfile

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    fbeta_score
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
PRED_DIR = DATA_DIR / "predictions"

MODEL_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", PROJECT_ROOT / "backend" / "models"))
for d in [PRED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

F_TRAIN = Path(os.getenv("TRAIN_TABLE", TRAIN_DIR / "personal_features_train_latest.csv"))
F_INFER = Path(os.getenv("INFER_TABLE", TRAIN_DIR / "personal_features_infer_latest.csv"))

TS = datetime.now(KST).strftime("%Y%m%d%H%M%S")

WEAR_SOFT = ["hr_bpm", "stress_0_1", "wearing", "reported", "symptom_score"]
WEATHER   = ["wbgt_c", "hi_c", "hours_wbgt_ge28_last6h", "temp_c", "rh_pct"]
PROFILE   = ["adherence", "hr_base", "fitness", "vulnerability"]
TIME_FEAT = ["hour", "dow"]
USE_COLS  = WEAR_SOFT + WEATHER + PROFILE + TIME_FEAT

TARGET_COL_ENV = os.getenv("TARGET_COL", "").strip()  
DEFAULT_TARGET_ORDER = ["has_patient", "hard_label"] 

VAL_SIZE = float(os.getenv("VAL_SIZE", "0.2"))              
MIN_VAL_POS = int(os.getenv("MIN_VAL_POS", "5"))              
RANDOM_SEED = int(os.getenv("SEED", "2025"))
SCALE_POS_WEIGHT = os.getenv("SCALE_POS_WEIGHT", "")        
UNDER_SAMPLE_NEG_RATIO = os.getenv("UNDER_SAMPLE_NEG_RATIO", "")  
TOPN = int(os.getenv("INFER_TOPN", "20"))                    
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

def _nan_guard(name, arr_like):
    if isinstance(arr_like, pd.DataFrame):
        if not np.isfinite(arr_like.values).all():
            bad_cols = [c for c in arr_like.columns if not np.isfinite(arr_like[c].values).all()]
            raise ValueError(f"{name}: non-finite values remain in columns {bad_cols}")
    else:
        if not np.isfinite(arr_like).all():
            raise ValueError(f"{name}: non-finite values remain.")

def _safe_report(y_true, y_prob, prefix="VAL", thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= thr).astype(int)

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
    elif n_pos == 0:
        auc, ap = float("nan"), 0.0
    else:  # n_neg == 0
        auc, ap = float("nan"), 1.0

    print(f"[{prefix}] ROC-AUC={auc:.4f}  AP={ap:.4f}  @thr={thr:.3f}  (pos={n_pos}, neg={n_neg})")
    print(f"[{prefix}] Report:\n", classification_report(y_true, y_pred, digits=3, zero_division=0))

def _tune_threshold_f2(y_true, y_prob):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    best_thr, best_f2 = 0.5, -1.0
    for thr in np.linspace(0.0, 1.0, 501):
        pred = (y_prob >= thr).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr = f2, thr
    return float(best_thr), float(best_f2)

# -------------------------
# Robust preprocessing
# -------------------------
def safe_fit_preprocess(df: pd.DataFrame, feature_cols: list):

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    medians = {}
    for c in X.columns:
        med = np.nanmedian(X[c].values)
        if np.isnan(med):
            med = 0.0
        medians[c] = float(med)
        X[c] = X[c].fillna(med)

    stds = X.std(axis=0, ddof=0)
    cols_keep = [c for c in X.columns if np.isfinite(stds[c]) and stds[c] > 0.0]
    if len(cols_keep) == 0:
        raise SystemExit("All features have zero variance after imputation. Check your data.")

    Xk = X[cols_keep].copy()
    _nan_guard("TRAIN_X_after_impute", Xk)

    scaler = StandardScaler()
    Xk_sc = scaler.fit_transform(Xk)
    _nan_guard("TRAIN_X_after_scaler", Xk_sc)

    return {"cols_keep": cols_keep, "medians": medians, "scaler": scaler}

def apply_preprocess(df: pd.DataFrame, preproc: dict):
    cols_keep = preproc["cols_keep"]
    medians   = preproc["medians"]
    scaler    = preproc["scaler"]

    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan

    X = df[cols_keep].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        X[c] = df[c].fillna(medians.get(c, 0.0))

    _nan_guard("APPLY_X_after_impute", X)
    X_sc = scaler.transform(X.values)
    _nan_guard("APPLY_X_after_scaler", X_sc)
    return X_sc

def apply_preprocess_noscale(df: pd.DataFrame, preproc: dict):
    cols_keep = preproc["cols_keep"]
    medians   = preproc["medians"]
    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan
    X = df[cols_keep].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in cols_keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        X[c] = df[c].fillna(medians.get(c, 0.0))
    _nan_guard("APPLY_XGB_X_after_impute", X)
    return X.values

# -------------------------
# 1) Load
# -------------------------
def _load_tables():
    _require(F_TRAIN, "train table")
    train = pd.read_csv(F_TRAIN)

    # Target 선택
    target_col = TARGET_COL_ENV if TARGET_COL_ENV else None
    if target_col and target_col not in train.columns:
        raise SystemExit(f"TARGET_COL={target_col} not found in train table.")
    if target_col is None:
        for cand in DEFAULT_TARGET_ORDER:
            if cand in train.columns:
                target_col = cand
                break
        if target_col is None:
            raise SystemExit(f"Train table must contain one of {DEFAULT_TARGET_ORDER} or set TARGET_COL env.")

    # dt_hour 처리
    if "dt_hour" in train.columns:
        train["dt_hour"] = _to_dt(train["dt_hour"])
        train = train.dropna(subset=["dt_hour", target_col]).copy()
    else:
        train = train.dropna(subset=[target_col]).copy()

    num_cols = _select_existing(train, USE_COLS)
    if len(num_cols) == 0:
        raise SystemExit("No usable feature columns found. Check USE_COLS and input tables.")
    train = _safe_float(train, num_cols)

    _require(F_INFER, "infer table")
    infer = pd.read_csv(F_INFER)
    if "dt_hour" in infer.columns:
        infer["dt_hour"] = _to_dt(infer["dt_hour"])
    infer = _safe_float(infer, _select_existing(infer, USE_COLS))
    return train, infer, num_cols, target_col

train, infer, num_cols, TARGET_COL = _load_tables()
print(f"[INFO] TARGET_COL = {TARGET_COL}")

# -------------------------
# 2) Train/Val split (time-based holdout + MIN_VAL_POS 보장 시도)
# -------------------------
if "dt_hour" in train.columns:
    train_sorted = train.sort_values("dt_hour")
else:
    train_sorted = train.copy()

cut_idx = int(len(train_sorted) * (1.0 - VAL_SIZE))
cut_idx = max(1, min(len(train_sorted)-1, cut_idx))  # guard

y_all = train_sorted[TARGET_COL].astype(int).values
pos_after = int(y_all[cut_idx:].sum())

if pos_after < MIN_VAL_POS:
    found = False
    for new_cut in range(cut_idx-1, -1, -1):
        if int(y_all[new_cut:].sum()) >= MIN_VAL_POS:
            cut_idx = new_cut
            found = True
            break
    if not found:
        print(f"[WARN] Could not secure MIN_VAL_POS={MIN_VAL_POS} in validation. Proceeding with pos={pos_after}.")

train_df = train_sorted.iloc[:cut_idx].copy()
val_df   = train_sorted.iloc[cut_idx:].copy()

y_tr = train_df[TARGET_COL].astype(int).values
y_val = val_df[TARGET_COL].astype(int).values

print("[SPLIT] cut_idx:", cut_idx,
      "| train len:", len(train_df), "pos:", int(y_tr.sum()),
      "| val len:", len(val_df), "pos:", int(y_val.sum()))

# -------------------------
# (선택) 간단 언더샘플링: 음성:양성 = R:1로 맞춤
# -------------------------
def _undersample_negatives(df, y, ratio_R=5, seed=RANDOM_SEED):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return df, y
    keep_neg = min(len(neg_idx), int(ratio_R * len(pos_idx)))
    rng = np.random.default_rng(seed)
    kept_neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False)
    kept_idx = np.concatenate([pos_idx, kept_neg_idx])
    kept_idx.sort()
    return df.iloc[kept_idx].copy(), y[kept_idx]

if UNDER_SAMPLE_NEG_RATIO.strip():
    R = max(1, int(float(UNDER_SAMPLE_NEG_RATIO)))
    print(f"[INFO] Apply negative undersampling: ratio {R}:1 (neg:pos)")
    train_df, y_tr = _undersample_negatives(train_df, y_tr, ratio_R=R, seed=RANDOM_SEED)

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

# 0.5 기준 + F2 최적 임계치
_safe_report(y_val, p_val_lr, prefix="VAL-LogReg@0.5", thr=0.5)
best_thr_lr, best_f2_lr = _tune_threshold_f2(y_val, p_val_lr)
print(f"[VAL-LogReg] Best F2 threshold = {best_thr_lr:.3f} (F2={best_f2_lr:.4f})")
_safe_report(y_val, p_val_lr, prefix="VAL-LogReg@bestF2", thr=best_thr_lr)

# Save LR artifacts
dump({"model": logreg, "preproc": preproc, "features_all": num_cols, "threshold": best_thr_lr},
     MODEL_DIR / f"logreg_personal_{TS}.joblib")
dump({"model": logreg, "preproc": preproc, "features_all": num_cols, "threshold": best_thr_lr},
     MODEL_DIR / "logreg_personal_latest.joblib")

# (B) XGBoost (optional)
if HAS_XGB:
    pos_weight = (len(y_tr) - int(y_tr.sum())) / max(1, int(y_tr.sum()))
    if SCALE_POS_WEIGHT.strip() != "":
        pos_weight = float(SCALE_POS_WEIGHT)

    X_tr_xgb = apply_preprocess_noscale(train_df, preproc)
    X_val_xgb = apply_preprocess_noscale(val_df, preproc)

    xgb = XGBClassifier(
        n_estimators=500,
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

    _safe_report(y_val, p_val_xgb, prefix="VAL-XGB@0.5", thr=0.5)
    best_thr_xgb, best_f2_xgb = _tune_threshold_f2(y_val, p_val_xgb)
    print(f"[VAL-XGB] Best F2 threshold = {best_thr_xgb:.3f} (F2={best_f2_xgb:.4f})")
    _safe_report(y_val, p_val_xgb, prefix="VAL-XGB@bestF2", thr=best_thr_xgb)

    dump({"model": xgb, "preproc": preproc, "features_all": num_cols, "threshold": best_thr_xgb},
         MODEL_DIR / f"xgb_personal_{TS}.joblib")
    dump({"model": xgb, "preproc": preproc, "features_all": num_cols, "threshold": best_thr_xgb},
         MODEL_DIR / "xgb_personal_latest.joblib")
else:
    print("[WARN] xgboost not installed; skipping XGB model.")

# -------------------------
# 5) Save validation predictions
# -------------------------
if "dt_hour" in val_df.columns:
    cols_core = ["user_id", "adm_cd2", "dt_hour", TARGET_COL]
else:
    val_df["dt_hour"] = pd.NaT
    cols_core = ["user_id", "adm_cd2", "dt_hour", TARGET_COL]

val_out = val_df[cols_core].copy()
val_out["p_logreg"] = p_val_lr
if HAS_XGB:
    val_out["p_xgb"] = p_val_xgb

val_path_ts = PRED_DIR / f"val_predictions_{TS}.csv"
val_latest  = PRED_DIR / "val_predictions_latest.csv"
val_out.to_csv(val_path_ts, index=False)
copyfile(val_path_ts, val_latest)
print(f"[SAVE] {val_path_ts.name}  (also wrote -> {val_latest.name})")

# -------------------------
# 6) Inference on 'now' table
# -------------------------
X_inf_sc = apply_preprocess(infer, preproc)
infer["risk_logreg"] = logreg.predict_proba(X_inf_sc)[:, 1]

if HAS_XGB:
    X_inf_xgb = apply_preprocess_noscale(infer, preproc)
    infer["risk_xgb"] = xgb.predict_proba(X_inf_xgb)[:, 1]

# risk_any = 사용 가능한 위험도 평균
cand_cols = [c for c in ["risk_logreg", "risk_xgb", "risk_score", "risk_heat", "risk"] if c in infer.columns]
infer["risk_any"] = infer[cand_cols].mean(axis=1, skipna=True) if len(cand_cols) > 0 else np.nan

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
