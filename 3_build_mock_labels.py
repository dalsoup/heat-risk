# build_mock_labels.py
# 동×시간 가상 온열환자수 라벨 생성 (안전 가드/스케일링 포함)

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# -------------------------
# Config
# -------------------------
KST = timezone(timedelta(hours=9))

FEATURES_PATH = Path("data/features/hourly_features_realtime_latest.csv")
OUT_DIR = Path("data/labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("MOCK_SEED", "2025"))

# 스케일 파라미터 (10만 명당, 시간당 환자수)
BASE_RATE_PER_100K_PER_HOUR = float(os.getenv("BASE_RATE_PER_100K_PER_HOUR", "0.02"))
# WBGT가 최댓값(정규화 1)에 도달했을 때 추가되는 10만 명당 환자수(상한)
WBGT_UPLIFT_PER_100K_MAX = float(os.getenv("WBGT_UPLIFT_PER_100K_MAX", "0.15"))

# 안전 가드
LAMBDA_MAX = float(os.getenv("MOCK_LAMBDA_MAX", "10.0"))  # 포아송 기대값 상한
MIN_POP = float(os.getenv("MIN_POP", "1000"))              # 비정상적으로 작은 인구 보정
DEFAULT_POP = float(os.getenv("DEFAULT_POP", "100000"))    # 인구 컬럼 없을 때 기본값

# (선택) 전체 평균 캘리브레이션: 목표 시간당 평균 환자수
TARGET_MEAN_CASES_PER_HOUR = os.getenv("TARGET_MEAN_CASES_PER_HOUR", "")
TARGET_MEAN_CASES_PER_HOUR = float(TARGET_MEAN_CASES_PER_HOUR) if TARGET_MEAN_CASES_PER_HOUR else None

# -------------------------
# Helpers
# -------------------------
def _ts():
    return datetime.now(KST).strftime("%Y%m%d%H%M%S")

def minmax01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x)
    y = (x - mn) / (mx - mn)
    return np.clip(y, 0.0, 1.0)

def finite_or(x, fill=0.0):
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = fill
    return x

def q(arr, p):
    return np.nanpercentile(arr, p)

# -------------------------
# Load features
# -------------------------
if not FEATURES_PATH.exists():
    raise SystemExit("Missing realtime features. Run build_features_hourly.py first.")

feat = pd.read_csv(FEATURES_PATH)
req = {"adm_cd2", "dt_hour", "wbgt_c"}
missing = req - set(feat.columns)
if missing:
    raise SystemExit(f"Missing columns in features: {missing}")

feat["dt_hour"] = pd.to_datetime(feat["dt_hour"], errors="coerce")
feat = feat.dropna(subset=["dt_hour"]).copy()
feat = feat.sort_values(["adm_cd2", "dt_hour"]).reset_index(drop=True)

# 인구 열이 없으면 기본값, 너무 작으면 MIN_POP로 보정
pop_col = None
for c in ["인구수", "population", "pop"]:
    if c in feat.columns:
        pop_col = c
        break
if pop_col is None:
    feat["population"] = DEFAULT_POP
    pop_col = "population"

feat[pop_col] = finite_or(feat[pop_col], DEFAULT_POP)
feat.loc[feat[pop_col] < MIN_POP, pop_col] = MIN_POP

# -------------------------
# Build λ (10만 명당 스케일 → 절대 건수)
# -------------------------
rng = np.random.default_rng(RANDOM_SEED)

feat["wbgt_01"] = minmax01(feat["wbgt_c"])

# 10만 명당 기대 환자수 (시간당)
rate_per_100k = (
    BASE_RATE_PER_100K_PER_HOUR
    + WBGT_UPLIFT_PER_100K_MAX * feat["wbgt_01"].values
)
rate_per_100k = finite_or(rate_per_100k, 0.0)

# 절대 λ = rate_per_100k * (population / 100000)
lam = rate_per_100k * (feat[pop_col].values / 100000.0)
lam = finite_or(lam, 0.0)

# (선택) 전체 평균 캘리브레이션
if TARGET_MEAN_CASES_PER_HOUR is not None:
    cur_mean = float(np.nanmean(lam)) if lam.size else 0.0
    if np.isfinite(cur_mean) and cur_mean > 0:
        scale = TARGET_MEAN_CASES_PER_HOUR / cur_mean
        lam = lam * scale
        print(f"[CALIB] mean λ: {cur_mean:.4f} -> target {TARGET_MEAN_CASES_PER_HOUR:.4f} (scale {scale:.3f})")
    else:
        print("[CALIB] skipped (invalid current mean)")

# 안전 가드: 음수/비유한 제거 + 상한
lam = np.clip(finite_or(lam, 0.0), 0.0, LAMBDA_MAX)

# 디버그 통계
print("[LAMBDA] stats:",
      f"min={np.min(lam):.4f}",
      f"p50={q(lam,50):.4f}",
      f"p90={q(lam,90):.4f}",
      f"max={np.max(lam):.4f}",
      f"mean={np.mean(lam):.4f}",
)

# -------------------------
# Sample counts
# -------------------------
feat["patient_count"] = rng.poisson(lam)

labels = feat[["adm_cd2", "dt_hour", "patient_count"]].copy()

# -------------------------
# Save
# -------------------------
ts = _ts()
out_path = OUT_DIR / f"patients_hourly_{ts}.csv"
latest_path = OUT_DIR / "patients_hourly_latest.csv"

labels.to_csv(out_path, index=False)
labels.to_csv(latest_path, index=False)

print("[SAVE]", out_path.name, "and", latest_path.name)
print("[INFO] rows:", len(labels))
print(labels.head())
