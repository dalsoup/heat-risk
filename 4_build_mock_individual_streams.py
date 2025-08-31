# build_mock_individual_streams.py
# 개인 단위 웨어러블/자가신고 가상 시계열 생성
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# -------------------------
# Config
# -------------------------
KST = timezone(timedelta(hours=9))

REALTIME_FEATURES = Path("data/features/hourly_features_realtime_latest.csv")
OUT_DIR = Path(os.getenv("LABELS_OUT_DIR", "data/labels"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 인구/스트림 스케일 파라미터 (환경변수로 조정 가능)
NUM_USERS_PER_DONG = int(os.getenv("MOCK_USERS_PER_DONG", "5"))   # 동별 유저 수 (기본 5)
MAX_USERS_TOTAL    = int(os.getenv("MOCK_MAX_USERS", "20000"))    # 전체 상한 (안전)
RANDOM_SEED        = int(os.getenv("MOCK_SEED", "2025"))

# 웨어러블 장치 착용/결측 모델
ADHERENCE_MEAN = float(os.getenv("MOCK_ADHERENCE_MEAN", "0.85"))  # 장치 착용 평균 확률
ADHERENCE_STD  = float(os.getenv("MOCK_ADHERENCE_STD",  "0.07"))
WEAR_NOISE     = float(os.getenv("MOCK_WEAR_NOISE",     "0.08"))  # stress 노이즈
WEAR_AR1       = float(os.getenv("MOCK_WEAR_AR1",       "0.7"))   # 개인 AR(1) 관성
HR_BASE_MEAN   = float(os.getenv("MOCK_HR_BASE_MEAN",   "70.0"))  # 개인 기저 심박
HR_BASE_STD    = float(os.getenv("MOCK_HR_BASE_STD",    "8.0"))

# 자가신고 확률 모델 (로지스틱)
SOFT_BASE      = float(os.getenv("MOCK_SOFT_BASE",      "-2.4"))  # 절편 (낮을수록 드묾)
SOFT_B_WBGT    = float(os.getenv("MOCK_SOFT_B_WBGT",    "2.2"))
SOFT_B_DIURNAL = float(os.getenv("MOCK_SOFT_B_DIURNAL", "0.6"))
SOFT_B_WEEKEND = float(os.getenv("MOCK_SOFT_B_WEEKEND", "0.25"))
SOFT_USER_HETEROGENEITY = float(os.getenv("MOCK_SOFT_USER_HET", "0.7"))  # 개인 성향 분산
SYMPTOM_SCALE  = float(os.getenv("MOCK_SYMPTOM_SCALE",  "7.0"))   # 증상 점수 스케일 상한 (0..10 중)

# (선택) 시뮬레이션 기간 제한 (최근 N일만)
RECENT_DAYS = os.getenv("MOCK_RECENT_DAYS", None)
if RECENT_DAYS not in (None, ""):
    RECENT_DAYS = int(RECENT_DAYS)
else:
    RECENT_DAYS = None

# -------------------------
# Helpers
# -------------------------
def _ts() -> str:
    return datetime.now(KST).strftime("%Y%m%d%H%M%S")

def _ensure_cols(df: pd.DataFrame, cols: set):
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

def _minmax01(x):
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x); mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x)
    y = (x - mn) / (mx - mn)
    return np.clip(y, 0.0, 1.0)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# -------------------------
# Load realtime features
# -------------------------
if not REALTIME_FEATURES.exists():
    raise SystemExit("Missing realtime features: data/features/hourly_features_realtime_latest.csv. "
                     "Run build_features_hourly.py first.")

feat = pd.read_csv(REALTIME_FEATURES)
_req = {"adm_cd2", "adm_nm", "dt_hour", "wbgt_c", "hi_c"}
_ensure_cols(feat, _req)

feat["dt_hour"] = pd.to_datetime(feat["dt_hour"], errors="coerce")
feat = feat.dropna(subset=["dt_hour", "adm_cd2"]).copy()

if feat.empty:
    raise SystemExit("Realtime features are empty.")

# (선택) 최근 N일만
if RECENT_DAYS:
    cutoff = feat["dt_hour"].max() - pd.Timedelta(days=RECENT_DAYS)
    feat = feat[feat["dt_hour"] >= cutoff].copy()

# 시간 파생
feat["hour"] = feat["dt_hour"].dt.hour
feat["dow"]  = feat["dt_hour"].dt.dayofweek
# diurnal: 14~17시 피크가 1에 가깝도록
diurnal = 0.5 + 0.5 * np.cos((feat["hour"] - 15) / 24.0 * 2 * np.pi)
feat["diurnal_01"] = _minmax01(1.0 - diurnal)

# 정규화된 열지표
feat["wbgt_01"] = _minmax01(feat["wbgt_c"])
feat["hi_01"]   = _minmax01(feat["hi_c"])
feat["is_weekend"] = (feat["dow"] >= 5).astype(int)

# 동 목록
dong_df = feat[["adm_cd2", "adm_nm"]].drop_duplicates().reset_index(drop=True)
dong_df["n_users"] = NUM_USERS_PER_DONG

# 전체 유저 수 상한 적용
total_users = int(dong_df["n_users"].sum())
if total_users > MAX_USERS_TOTAL:
    scale = MAX_USERS_TOTAL / total_users
    dong_df["n_users"] = np.maximum(1, (dong_df["n_users"] * scale).astype(int))
    print(f"[WARN] users capped: {total_users} -> {int(dong_df['n_users'].sum())}")

# -------------------------
# Generate users (meta)
# -------------------------
rng = np.random.default_rng(RANDOM_SEED)

user_rows = []
for _, r in dong_df.iterrows():
    adm = str(r["adm_cd2"])
    n   = int(r["n_users"])
    # 개인 속성 샘플
    adherence = np.clip(rng.normal(ADHERENCE_MEAN, ADHERENCE_STD, size=n), 0.4, 0.99)
    hr_base   = rng.normal(HR_BASE_MEAN, HR_BASE_STD, size=n)
    fitness   = rng.normal(0.0, 0.4, size=n)   # 체력(+)/비만(-) 등 임의 지표
    vuln      = rng.normal(0.0, 0.6, size=n)   # 취약성(+)
    soft_bias = rng.normal(0.0, SOFT_USER_HETEROGENEITY, size=n)  # 자가신고 성향

    for i in range(n):
        user_id = f"{adm}_{i:04d}"
        user_rows.append({
            "user_id": user_id,
            "home_adm_cd2": adm,
            "home_adm_nm": r["adm_nm"],
            "adherence": float(adherence[i]),
            "hr_base": float(hr_base[i]),
            "fitness": float(fitness[i]),
            "vulnerability": float(vuln[i]),
            "selfreport_bias": float(soft_bias[i]),
        })

users = pd.DataFrame(user_rows)

# 저장 (유저 메타)
ts = _ts()
users_path = OUT_DIR / f"individual_users_{ts}.csv"
users_latest = OUT_DIR / "individual_users_latest.csv"
users.to_csv(users_path, index=False)
users.to_csv(users_latest, index=False)
print(f"[INFO] users generated: {len(users):,}")

# -------------------------
# Build per-user time series
# -------------------------
# 각 동별 시간축 준비
dong_time = feat[["adm_cd2", "dt_hour", "wbgt_01", "hi_01", "diurnal_01", "is_weekend"]].copy()

wear_stream = []
self_stream = []

for adm, gfeat in dong_time.groupby("adm_cd2"):
    gfeat = gfeat.sort_values("dt_hour").reset_index(drop=True)
    # 해당 동의 유저들
    u = users[users["home_adm_cd2"] == str(adm)].reset_index(drop=True)
    if u.empty:
        continue

    T = len(gfeat)
    M = len(u)

    # -------- 웨어러블 시뮬레이션 --------
    # 위험 시그널: wbgt/hi/diurnal 결합
    risk = 0.6 * gfeat["wbgt_01"].values + 0.4 * gfeat["hi_01"].values + 0.2 * gfeat["diurnal_01"].values
    risk = np.clip(risk, 0.0, 1.0)

    # 장치 착용 여부(시간×유저): 베르누이(adherence) + 주말/야간 소폭 하락
    adher = u["adherence"].values[None, :]  # 1×M
    weekend = gfeat["is_weekend"].values[:, None]
    hour = gfeat["dt_hour"].dt.hour.values[:, None]
    night = ((hour >= 0) & (hour <= 6)).astype(int)
    adher_t = np.clip(adher - 0.05 * weekend - 0.07 * night, 0.2, 0.99)

    wear_on = rng.random((T, M)) < adher_t  # True/False

    # HR/Stress AR(1)
    hr = np.zeros((T, M), dtype=float)
    stress = np.zeros((T, M), dtype=float)

    hr_base = u["hr_base"].values[None, :]
    fitness = u["fitness"].values[None, :]
    vuln    = u["vulnerability"].values[None, :]

    prev_hr = hr_base + 2.0 * rng.normal(0, 1.0, size=(1, M))
    prev_st = 0.2 + 0.2 * rng.random(size=(1, M))

    for t in range(T):
        # 위험이 HR에 미치는 영향 (취약성/체력 가중)
        risk_push = (1.0 + 0.5 * vuln - 0.3 * fitness) * (5.0 * risk[t])
        eps_hr = rng.normal(0.0, 2.0, size=(1, M))
        eps_st = rng.normal(0.0, WEAR_NOISE, size=(1, M))
        now_hr = WEAR_AR1 * prev_hr + (1 - WEAR_AR1) * (hr_base + risk_push) + eps_hr
        now_st = WEAR_AR1 * prev_st + (1 - WEAR_AR1) * (0.2 + 0.8 * risk[t]) + eps_st

        # 장치 미착용 시간은 관측값 NaN 처리
        mask = wear_on[t:t+1, :]
        hr[t:t+1, :] = np.where(mask, now_hr, np.nan)
        stress[t:t+1, :] = np.where(mask, now_st, np.nan)

        prev_hr = now_hr
        prev_st = now_st

    # 결과 프레임(롱포맷)
    w = pd.DataFrame({
        "dt_hour": np.repeat(gfeat["dt_hour"].values, M),
        "user_id": np.tile(u["user_id"].values, T),
        "adm_cd2": str(adm),
        "wearing": wear_on.reshape(-1).astype(int),
        "hr_bpm": hr.reshape(-1),
        "stress_0_1": np.clip(stress.reshape(-1), 0.0, 1.0),
    })
    wear_stream.append(w)

    # -------- 자가신고 시뮬레이션 --------
    # 로지스틱 확률: σ(β0 + β1 * risk + β2 * diurnal + β3 * weekend + 개인성향)
    soft_bias = u["selfreport_bias"].values[None, :]
    lin = (SOFT_BASE
           + SOFT_B_WBGT * risk[:, None]
           + SOFT_B_DIURNAL * gfeat["diurnal_01"].values[:, None]
           + SOFT_B_WEEKEND * gfeat["is_weekend"].values[:, None]
           + soft_bias)
    p = _sigmoid(lin)

    report = (rng.random((T, M)) < p).astype(int)
    # 증상 점수 (보고한 경우에만)
    sym = np.where(report == 1,
                   np.clip(SYMPTOM_SCALE * (risk[:, None] + 0.3 * rng.normal(0, 0.3, size=(T, M))), 0.0, 10.0),
                   np.nan)

    s = pd.DataFrame({
        "dt_hour": np.repeat(gfeat["dt_hour"].values, M),
        "user_id": np.tile(u["user_id"].values, T),
        "adm_cd2": str(adm),
        "reported": report.reshape(-1).astype(int),
        "symptom_score": sym.reshape(-1),
    })
    self_stream.append(s)

# 합치기
wear_df = pd.concat(wear_stream, ignore_index=True) if wear_stream else pd.DataFrame()
self_df = pd.concat(self_stream, ignore_index=True) if self_stream else pd.DataFrame()

if wear_df.empty or self_df.empty:
    raise SystemExit("No individual streams generated. Check parameters or features.")

# 저장
ts = _ts()
paths = {
    "users": OUT_DIR / f"individual_users_{ts}.csv",
    "wear":  OUT_DIR / f"individual_wearable_{ts}.csv",
    "self":  OUT_DIR / f"individual_selfreport_{ts}.csv",
}
latest = {
    "users": OUT_DIR / "individual_users_latest.csv",
    "wear":  OUT_DIR / "individual_wearable_latest.csv",
    "self":  OUT_DIR / "individual_selfreport_latest.csv",
}

# users는 위에서 이미 저장했으나 동일한 경로 사용 유지
wear_df.to_csv(paths["wear"], index=False);  wear_df.to_csv(latest["wear"], index=False)
self_df.to_csv(paths["self"], index=False);  self_df.to_csv(latest["self"], index=False)

print("[SAVE]")
print(" -", paths["users"].name, "and", latest["users"].name)
print(" -", paths["wear"].name,  "and", latest["wear"].name)
print(" -", paths["self"].name,  "and", latest["self"].name)
print(f"[INFO] wearable rows: {len(wear_df):,}  self-report rows: {len(self_df):,}")
