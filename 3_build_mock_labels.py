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
WBGT_UPLIFT_PER_100K_MAX    = float(os.getenv("WBGT_UPLIFT_PER_100K_MAX", "0.15"))

# 안전 가드
LAMBDA_MAX  = float(os.getenv("MOCK_LAMBDA_MAX", "10.0"))
MIN_POP     = float(os.getenv("MIN_POP", "1000"))
DEFAULT_POP = float(os.getenv("DEFAULT_POP", "100000"))

# (선택) 전체 평균 캘리브레이션: 목표 시간당 평균 환자수
TARGET_MEAN_CASES_PER_HOUR = os.getenv("TARGET_MEAN_CASES_PER_HOUR", "")
TARGET_MEAN_CASES_PER_HOUR = float(TARGET_MEAN_CASES_PER_HOUR) if TARGET_MEAN_CASES_PER_HOUR else None

# (선택) 양성비(>0) 캘리브레이션: 전체에서 사건 발생 확률 타깃
TARGET_POSITIVE_RATE = os.getenv("TARGET_POSITIVE_RATE", "")
TARGET_POSITIVE_RATE = float(TARGET_POSITIVE_RATE) if TARGET_POSITIVE_RATE else None

# Zero-Inflation Gate 로짓 가중치
LOGIT_BASE     = float(os.getenv("EVENT_LOGIT_BASE", "-3.0"))
W_WBGT         = float(os.getenv("EVENT_LOGIT_W_WBGT", "3.0"))
W_HOTSPELL     = float(os.getenv("EVENT_LOGIT_W_HOTSPELL", "1.5"))
W_TOD          = float(os.getenv("EVENT_LOGIT_W_TOD", "1.2"))
W_HUM          = float(os.getenv("EVENT_LOGIT_W_HUM", "0.6"))
W_SPATIAL      = float(os.getenv("EVENT_LOGIT_W_SPATIAL", "0.8"))
WEEKEND_FACTOR = float(os.getenv("WEEKEND_FACTOR", "0.85"))

# 시간대 피크 설정
TOD_PEAK_HOUR  = int(os.getenv("TOD_PEAK_HOUR", "15"))
TOD_STD_HOURS  = float(os.getenv("TOD_STD_HOURS", "4.0"))

# 핫스펠 기준
BASE_WBGT = float(os.getenv("BASE_WBGT_FOR_EXPOSURE", "28.0"))

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

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def stable_hash_to_normal(s, rng):
    h = abs(hash(str(s))) % (10**6)
    rng_local = np.random.default_rng(rng.integers(1, 10**9) + h)
    return rng_local.normal(loc=0.0, scale=1.0)

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

# 인구 열 보정
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

rng = np.random.default_rng(RANDOM_SEED)

# -------------------------
# Derived features
# -------------------------
feat["wbgt_01"] = minmax01(feat["wbgt_c"])

feat["wbgt_excess"] = np.clip(feat["wbgt_c"] - BASE_WBGT, 0.0, None)
feat["hotspell_6h"] = (
    feat.groupby("adm_cd2", observed=True)["wbgt_excess"]
        .rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True)
)
feat["hotspell_01"] = minmax01(feat["hotspell_6h"])

hour = feat["dt_hour"].dt.hour.values
tod_shape = np.exp(-((hour - TOD_PEAK_HOUR) ** 2) / (2.0 * (TOD_STD_HOURS ** 2)))
feat["tod_01"] = minmax01(tod_shape)

hum_col = None
for c in ["rel_humidity", "humidity", "hum", "습도"]:
    if c in feat.columns:
        hum_col = c
        break
feat["hum_01"] = minmax01(feat[hum_col].values) if hum_col else 0.0

dow = feat["dt_hour"].dt.dayofweek.values  # 0=월 ... 6=일
is_weekend = (dow >= 5).astype(float)

dong_uniques = feat["adm_cd2"].unique()
spatial_map = {d: stable_hash_to_normal(d, rng) for d in dong_uniques}
feat["spatial_z"] = feat["adm_cd2"].map(spatial_map).astype(float)

# -------------------------
# Zero-Inflation Gate
# -------------------------
logit = (
    LOGIT_BASE
    + W_WBGT    * feat["wbgt_01"].values
    + W_HOTSPELL* feat["hotspell_01"].values
    + W_TOD     * feat["tod_01"].values
    + W_HUM     * feat["hum_01"].values
    + W_SPATIAL * feat["spatial_z"].values
)
if WEEKEND_FACTOR != 1.0:
    logit = logit + np.log(np.where(is_weekend > 0, WEEKEND_FACTOR, 1.0))

p_event = sigmoid(logit)

# -------------------------
# λ (강도)
# -------------------------
severity_boost = 1.0 + 2.0 * (feat["wbgt_01"].values ** 2)
rate_per_100k = (
    BASE_RATE_PER_100K_PER_HOUR
    + WBGT_UPLIFT_PER_100K_MAX * feat["wbgt_01"].values
) * severity_boost

rate_per_100k = finite_or(rate_per_100k, 0.0)
lam = rate_per_100k * (feat[pop_col].values / 100000.0)
lam = finite_or(lam, 0.0)

# -------------------------
# 캘리브레이션
# -------------------------
# (1) 양성비: p_event 평균을 TARGET_POSITIVE_RATE로 맞춤
if TARGET_POSITIVE_RATE is not None:
    cur_pos = float(np.nanmean(p_event)) if p_event.size else 0.0
    if 0.0 < cur_pos < 1.0 and 0.0 < TARGET_POSITIVE_RATE < 1.0:
        shift = np.log(TARGET_POSITIVE_RATE / (1.0 - TARGET_POSITIVE_RATE)) - \
                np.log(cur_pos / (1.0 - cur_pos))
        p_event = sigmoid(logit + shift)
        print(f"[CALIB-POS] mean p_event: {cur_pos:.4f} -> target {TARGET_POSITIVE_RATE:.4f} (shift {shift:+.3f})")
    else:
        print("[CALIB-POS] skipped (invalid current/target rate)")

# (2) 평균 λ
if TARGET_MEAN_CASES_PER_HOUR is not None:
    cur_mean = float(np.nanmean(lam)) if lam.size else 0.0
    if np.isfinite(cur_mean) and cur_mean > 0:
        scale = TARGET_MEAN_CASES_PER_HOUR / cur_mean
        lam = lam * scale
        print(f"[CALIB-LAM] mean λ: {cur_mean:.4f} -> target {TARGET_MEAN_CASES_PER_HOUR:.4f} (scale {scale:.3f})")
    else:
        print("[CALIB-LAM] skipped (invalid current mean)")

# 안전 가드
lam = np.clip(finite_or(lam, 0.0), 0.0, LAMBDA_MAX)

# -------------------------
# 샘플링 (Zero-Truncated Poisson)
# -------------------------
rng_u = rng.random(size=len(feat))
will_happen = (rng_u < p_event).astype(int)   

counts = np.zeros(len(feat), dtype=int)
pos_idx = np.where(will_happen == 1)[0]
if len(pos_idx) > 0:
    sampled = rng.poisson(lam[pos_idx])
    sampled[sampled <= 0] = 1  
    counts[pos_idx] = sampled

has_patient = will_happen.astype(int)

# -------------------------
# 디버그 통계
# -------------------------
print("[EVENT] p_event stats:",
      f"mean={np.mean(p_event):.4f}",
      f"p90={q(p_event, 90):.4f}",
      f"max={np.max(p_event):.4f}")
print("[LAMBDA] stats:",
      f"min={np.min(lam):.4f}",
      f"p50={q(lam,50):.4f}",
      f"p90={q(lam,90):.4f}",
      f"max={np.max(lam):.4f}",
      f"mean={np.mean(lam):.4f}")
print("[LABEL] counts>0 rate:",
      f"{np.mean(counts>0):.4f}",
      "| mean(counts)=", f"{np.mean(counts):.4f}",
      "| max(counts)=", np.max(counts))

# -------------------------
# Save
# -------------------------
labels = feat[["adm_cd2", "dt_hour"]].copy()
labels["patient_count"] = counts
labels["has_patient"]   = has_patient

ts = _ts()
out_path    = OUT_DIR / f"patients_hourly_{ts}.csv"
latest_path = OUT_DIR / "patients_hourly_latest.csv"

labels.to_csv(out_path, index=False)
labels.to_csv(latest_path, index=False)

print("[SAVE]", out_path.name, "and", latest_path.name)
print("[INFO] rows:", len(labels))
print(labels.head())
