import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# -------------------------
# Config
# -------------------------
KST = timezone(timedelta(hours=9))

DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
OUT_DIR = DATA_DIR / "train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 입력 파일 경로 (환경변수로 오버라이드 가능)
F_WEATHER = Path(os.getenv("PERSONAL_F_WEATHER", FEAT_DIR / "hourly_features_realtime_latest.csv"))
F_USERS   = Path(os.getenv("PERSONAL_F_USERS",   LABEL_DIR / "individual_users_latest.csv"))
F_WEAR    = Path(os.getenv("PERSONAL_F_WEAR",    LABEL_DIR / "individual_wearable_latest.csv"))
F_SELF    = Path(os.getenv("PERSONAL_F_SELF",    LABEL_DIR / "individual_selfreport_latest.csv"))
F_STATIC  = Path(os.getenv("PERSONAL_F_STATIC",  DATA_DIR / "static.csv"))  
F_PATIENT = Path(os.getenv("PERSONAL_F_PATIENT", LABEL_DIR / "patients_hourly_latest.csv")) 

# 출력 파일
TS = datetime.now(KST).strftime('%Y%m%d%H%M%S')
OUT_TRAIN_TS = OUT_DIR / f"personal_features_train_{TS}.csv"
OUT_TRAIN    = OUT_DIR / "personal_features_train_latest.csv"
OUT_INFER_TS = OUT_DIR / f"personal_features_infer_{TS}.csv"
OUT_INFER    = OUT_DIR / "personal_features_infer_latest.csv"

# 파라미터
RECENT_DAYS = os.getenv("PERSONAL_RECENT_DAYS", "")  
RECENT_DAYS = int(RECENT_DAYS) if str(RECENT_DAYS).strip() != "" else None

# 라벨 이진화 임계 (patient_count > 임계 → 1)
PATIENT_POSITIVE_IF_GT = int(os.getenv("PATIENT_POSITIVE_IF_GT", "0")) 

# -------------------------
# Helpers
# -------------------------
def _require(path: Path, name: str):
    if not path.exists():
        raise SystemExit(f"Missing {name}: {path}")

def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def _minmax01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)

def _now_floor_naive_kst() -> pd.Timestamp:
    kst_now = datetime.now(KST).replace(minute=0, second=0, microsecond=0)
    return pd.Timestamp(kst_now.replace(tzinfo=None))

# -------------------------
# 0) Load inputs
# -------------------------
_require(F_WEATHER, "weather features (realtime)")
_require(F_USERS,   "user meta")
_require(F_WEAR,    "individual wearable")
_require(F_SELF,    "individual selfreport")

wth = pd.read_csv(F_WEATHER)
usr = pd.read_csv(F_USERS)
wrb = pd.read_csv(F_WEAR)
srf = pd.read_csv(F_SELF)

# Optional
stc = pd.read_csv(F_STATIC) if F_STATIC.exists() else pd.DataFrame()
pat = pd.read_csv(F_PATIENT) if F_PATIENT.exists() else pd.DataFrame()

# -------------------------
# 1) Basic cleaning / type casting
# -------------------------
for df, name in [(wth, "weather"), (wrb, "wearable"), (srf, "selfreport")]:
    if "dt_hour" not in df.columns:
        raise SystemExit(f"{name} is missing 'dt_hour' column.")
    df["dt_hour"] = _to_dt(df["dt_hour"])

# 키 컬럼 존재 체크
need_wth = {"adm_cd2", "dt_hour"}
need_wrb = {"user_id", "dt_hour", "adm_cd2"}
need_srf = {"user_id", "dt_hour", "adm_cd2"}
need_usr = {"user_id", "home_adm_cd2"}

for need, df, name in [
    (need_wth, wth, "weather"),
    (need_wrb, wrb, "wearable"),
    (need_srf, srf, "selfreport"),
    (need_usr, usr, "users"),
]:
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{name} missing columns: {missing}")

if RECENT_DAYS:
    cutoff = max(wth["dt_hour"].max(), wrb["dt_hour"].max(), srf["dt_hour"].max()) - pd.Timedelta(days=RECENT_DAYS)
    wth = wth[wth["dt_hour"] >= cutoff].copy()
    wrb = wrb[wrb["dt_hour"] >= cutoff].copy()
    srf = srf[srf["dt_hour"] >= cutoff].copy()

now_floor = _now_floor_naive_kst()
for name, df in [("weather", wth), ("wearable", wrb), ("selfreport", srf)]:
    df = df[df["dt_hour"] <= now_floor].copy()
    if name == "weather": wth = df
    if name == "wearable": wrb = df
    if name == "selfreport": srf = df

# -------------------------
# 2) Join wearable <-> selfreport (user_id, dt_hour, adm_cd2)
# -------------------------
ux = pd.merge(wrb, srf, on=["user_id", "dt_hour", "adm_cd2"], how="outer")

# -------------------------
# 3) Join weather features on (adm_cd2, dt_hour)
# -------------------------
weather_cols_keep = ["adm_cd2", "dt_hour", "wbgt_c", "hi_c", "hours_wbgt_ge28_last6h",
                     "temp_c", "rh_pct"]
weather_cols_keep = [c for c in weather_cols_keep if c in wth.columns]
wth_sl = wth[weather_cols_keep].drop_duplicates(["adm_cd2", "dt_hour"])

ux = pd.merge(ux, wth_sl, on=["adm_cd2", "dt_hour"], how="left")

# -------------------------
# 4) Join static on adm_cd2 (optional)
# -------------------------
if not stc.empty:
    if "adm_cd2" not in stc.columns:
        print("[WARN] static.csv has no adm_cd2. Skipped.")
    else:
        ux = pd.merge(ux, stc, on="adm_cd2", how="left")

# -------------------------
# 5) Join user meta (profile)
# -------------------------
usr_cols = ["user_id", "home_adm_cd2", "adherence", "hr_base", "fitness", "vulnerability"]
usr_cols = [c for c in usr_cols if c in usr.columns]
ux = pd.merge(ux, usr[usr_cols], on="user_id", how="left")

# 파생 시간 컬럼
ux["hour"] = ux["dt_hour"].dt.hour
ux["dow"]  = ux["dt_hour"].dt.dayofweek

# -------------------------
# 6) Attach labels from patients_hourly (has_patient & hard_label)
# -------------------------
if not pat.empty:
    # 표준화
    if "dt_hour" in pat.columns:
        pat["dt_hour"] = _to_dt(pat["dt_hour"])
    req_cols = {"adm_cd2", "dt_hour"}
    if not req_cols.issubset(pat.columns):
        print("[WARN] patients file lacks (adm_cd2, dt_hour). Skip labels.")
    else:
        keep_cols = ["adm_cd2", "dt_hour"]
        # 있을 수 있는 라벨/수치들
        if "patient_count" in pat.columns:
            keep_cols.append("patient_count")
        if "has_patient" in pat.columns:
            keep_cols.append("has_patient")
        pat_sl = pat[keep_cols].copy()

        # 결합
        ux = pd.merge(ux, pat_sl, on=["adm_cd2", "dt_hour"], how="left")

        # has_patient 생성/정리: 우선순위 = (컬럼 존재) else (patient_count > threshold)
        if "has_patient" in ux.columns:
            ux["has_patient"] = ux["has_patient"].fillna(0).astype(int)
        else:
            pc = ux["patient_count"] if "patient_count" in ux.columns else 0
            ux["has_patient"] = (pd.Series(pc).fillna(0) > PATIENT_POSITIVE_IF_GT).astype(int)

        # hard_label은 patient_count 기준(항상 보존)
        if "patient_count" in ux.columns:
            ux["hard_label"] = (ux["patient_count"].fillna(0) > PATIENT_POSITIVE_IF_GT).astype(int)
        else:
            # patient_count 없으면 has_patient로 대체 생성
            ux["hard_label"] = ux["has_patient"].astype(int)
else:
    print("[WARN] patients_hourly_latest.csv not found. No labels attached.")

# -------------------------
# 7) Cleanup / NA handling / sort
# -------------------------
# 기상 NaN 행 제거(학습 안정성)
drop_rows = ux["wbgt_c"].isna() | ux["hi_c"].isna()
if drop_rows.any():
    ux = ux[~drop_rows].copy()

# 정렬
ux = ux.sort_values(["user_id", "dt_hour"]).reset_index(drop=True)

# 라벨 요약(디버그)
if "has_patient" in ux.columns:
    print("[LABEL] has_patient pos:", int((ux["has_patient"]==1).sum()))
if "hard_label" in ux.columns:
    print("[LABEL] hard_label pos:", int((ux["hard_label"]==1).sum()))

# -------------------------
# 8) Save training table
# -------------------------
ux.to_csv(OUT_TRAIN_TS, index=False)
ux.to_csv(OUT_TRAIN,    index=False)
print(f"[SAVE] train table -> {OUT_TRAIN_TS.name} and {OUT_TRAIN.name}")
print(f"[INFO] train rows: {len(ux):,}  users: {ux['user_id'].nunique():,}")

# -------------------------
# 9) Make real-time inference table (one row per user at now_floor)
# -------------------------
now_floor = _now_floor_naive_kst()

# 현재 정각 행이 있으면 그대로, 없으면 각 user의 가장 최근 시각 사용(≤ now_floor)
ux_now = ux[ux["dt_hour"] == now_floor].copy()
if ux_now.empty:
    ux_recent = (
        ux[ux["dt_hour"] <= now_floor]
        .groupby("user_id", as_index=False)["dt_hour"].max()
        .rename(columns={"dt_hour": "dt_hour_max"})
    )
    ux_now = ux.merge(ux_recent, left_on=["user_id", "dt_hour"],
                      right_on=["user_id", "dt_hour_max"], how="inner")
    ux_now = ux_now.drop(columns=["dt_hour_max"])

ux_now.to_csv(OUT_INFER_TS, index=False)
ux_now.to_csv(OUT_INFER,    index=False)
print(f"[SAVE] infer table -> {OUT_INFER_TS.name} and {OUT_INFER.name}")
print(f"[INFO] infer rows: {len(ux_now):,}  users: {ux_now['user_id'].nunique():,}")

# 샘플 출력
print(ux_now.head(5))
