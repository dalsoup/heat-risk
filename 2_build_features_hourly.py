# build_features_hourly.py — realtime/hourly feature engineering (all dongs)
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# -------------------------
# Config
# -------------------------
KST = timezone(timedelta(hours=9))

PARSED_DIR = Path(os.getenv("PARSED_DIR", "data/weather_parsed"))
STATIC_PATH = Path(os.getenv("STATIC_CSV", "data/static.csv"))
OUT_DIR = Path(os.getenv("FEATURE_OUT_DIR", "data/features"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 입력 소스 (없으면 건너뜀)
SRC_FILES = [
    PARSED_DIR / "ncst_by_dong_latest.csv",
    PARSED_DIR / "ufc_by_dong_latest.csv",
    PARSED_DIR / "vfc_by_dong_latest.csv",
]

# 결측 채움 허용 시간(시간 단위). 최근 2시간 공백까지 ffill 허용.
FFILL_HOURS = int(os.getenv("FFILL_HOURS", "2"))

# 윈도우 길이(시간)
MA_WINDOWS = [3, 6, 12]   # 이동평균
THRESH_WINDOW = 6         # 임계이상 카운트 윈도우
LAG_1H = 1
LAG_24H = 24

# -------------------------
# Utils
# -------------------------
def _ts() -> str:
    return datetime.now(KST).strftime("%Y%m%d%H%M%S")

def _safe_read(p: Path) -> pd.DataFrame:
    if not p.exists():
        print(f"[WARN] missing: {p}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        return df
    except Exception as e:
        print(f"[WARN] failed reading {p}: {e}")
        return pd.DataFrame()

def _ensure_dt_col(df: pd.DataFrame) -> pd.DataFrame:
    if "dt" not in df.columns or df.empty:
        return df
    # 문자열/타임존 섞여 들어와도 안전하게 파싱
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    return df

# -------------------------
# Merge sources (priority: ncst > ufc > vfc)
# -------------------------
def _merge_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    pri_map = {"ncst": 3, "ufc": 2, "vfc": 1}

    tagged = []
    for f in frames:
        src = getattr(f, "source_name", None)
        if src is None:
            # 컬럼 힌트로 추정
            cols = set(f.columns)
            if {"T1H", "REH"}.issubset(cols):
                src = "ncst"
            elif "TMP" in cols or "PCP" in cols or "WSD" in cols:
                src = "ufc"
            else:
                src = "vfc"
        g = f.copy()
        g["__src_pri__"] = pri_map.get(src, 1)
        tagged.append(g)

    df = pd.concat(tagged, ignore_index=True)
    df = _ensure_dt_col(df)

    required = {"adm_cd2", "adm_nm", "dt"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Merged weather is missing columns: {missing}")

    # 우선순위로 정렬 후 (adm_cd2, dt) 고유화
    df = df.dropna(subset=["adm_cd2", "dt"])
    df = df.sort_values(["adm_cd2", "dt", "__src_pri__"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["adm_cd2", "dt"], keep="first")
    df = df.drop(columns=["__src_pri__"], errors="ignore")
    return df

# -------------------------
# Meteorology helpers
# -------------------------
def _choose_cols(df: pd.DataFrame):
    # 기온 후보
    for c in ["T1H", "TMP", "temp_c", "TEMP", "t"]:
        if c in df.columns:
            temp_col = c
            break
    else:
        temp_col = None

    # 습도 후보
    for c in ["REH", "RH", "rh_pct", "rh"]:
        if c in df.columns:
            rh_col = c
            break
    else:
        rh_col = None

    return temp_col, rh_col

def calc_heat_index_c(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    T = temp_c.astype(float)
    R = rh.astype(float).clip(lower=0, upper=100)
    HI = (
        -8.78469475556
        + 1.61139411 * T
        + 2.33854883889 * R
        - 0.14611605 * T * R
        - 0.012308094 * (T ** 2)
        - 0.0164248277778 * (R ** 2)
        + 0.002211732 * (T ** 2) * R
        + 0.00072546 * T * (R ** 2)
        - 0.000003582 * (T ** 2) * (R ** 2)
    )
    return HI

def approx_wbgt_shade_c(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    T = temp_c.astype(float)
    R = rh.astype(float).clip(0, 100)
    Twb = (
        T * np.arctan(0.151977 * np.sqrt(R + 8.313659))
        + np.arctan(T + R)
        - np.arctan(R - 1.676331)
        + 0.00391838 * (R ** 1.5) * np.arctan(0.023101 * R)
        - 4.686035
    )
    WBGT = 0.7 * Twb + 0.3 * T
    return WBGT

# -------------------------
# Build hourly base
# -------------------------
def make_hourly_base(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return merged

    temp_col, rh_col = _choose_cols(merged)
    if temp_col is None or rh_col is None:
        raise ValueError("Merged weather must include temperature and humidity columns (e.g., T1H/TMP and REH).")

    df = merged[["adm_cd2", "adm_nm", "dt", temp_col, rh_col]].copy()
    df = df.rename(columns={temp_col: "temp_c", rh_col: "rh_pct"})
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    # tz-aware → KST로 변환 후 naive로; tz-naive는 그대로
    try:
        tzinfo = df["dt"].dt.tz  # pandas가 전체 Series의 tz를 노출 (혼합이면 None일 수 있음)
    except Exception:
        tzinfo = None
    if tzinfo is not None:
        df["dt"] = df["dt"].dt.tz_convert(KST).dt.tz_localize(None)

    # 정각으로 스냅
    df["dt_hour"] = df["dt"].dt.floor("H")

    # 같은 시간대 중복은 평균으로 눌러서 안정화
    base = (
        df.groupby(["adm_cd2", "adm_nm", "dt_hour"], as_index=False)[["temp_c", "rh_pct"]]
          .mean()
          .sort_values(["adm_cd2", "dt_hour"])
    )

    # 파생: HI/WBGT
    base["hi_c"]   = calc_heat_index_c(base["temp_c"], base["rh_pct"])
    base["wbgt_c"] = approx_wbgt_shade_c(base["temp_c"], base["rh_pct"])

    return base

# -------------------------
# Fill gaps and expand to hourly grid
# -------------------------
def _fill_and_complete_hourly(base: pd.DataFrame, ffill_hours: int = FFILL_HOURS) -> pd.DataFrame:
    """
    동별 시간 축을 연속적(hourly)으로 만들고, 최대 ffill_hours까지 결측은 앞값 채움.
    """
    if base.empty:
        return base

    filled = []
    for adm, g in base.groupby("adm_cd2", as_index=False):
        g = g.sort_values("dt_hour")
        # 연속 시간 인덱스 만들기
        idx = pd.date_range(g["dt_hour"].min(), g["dt_hour"].max(), freq="H")
        hg = g.set_index("dt_hour").reindex(idx)
        hg.index.name = "dt_hour"
        # 앞값 채움 제한: 최근 ffill_hours시간까지만
        cols = ["temp_c", "rh_pct", "hi_c", "wbgt_c"]
        present = [c for c in cols if c in hg.columns]
        if present:
            hg[present] = hg[present].fillna(method="ffill", limit=ffill_hours)
        # 메타 복사
        hg["adm_cd2"] = adm
        # adm_nm은 가장 최근 값 사용
        hg["adm_nm"] = g["adm_nm"].iloc[-1]
        filled.append(hg.reset_index())

    out = pd.concat(filled, ignore_index=True)
    return out

# -------------------------
# Feature engineering (hourly)
# -------------------------
def add_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values(["adm_cd2", "dt_hour"])
    # 라그
    df["temp_lag1h"]  = df.groupby("adm_cd2")["temp_c"].shift(LAG_1H)
    df["rh_lag1h"]    = df.groupby("adm_cd2")["rh_pct"].shift(LAG_1H)
    df["hi_lag1h"]    = df.groupby("adm_cd2")["hi_c"].shift(LAG_1H)
    df["wbgt_lag1h"]  = df.groupby("adm_cd2")["wbgt_c"].shift(LAG_1H)

    df["temp_lag24h"] = df.groupby("adm_cd2")["temp_c"].shift(LAG_24H)
    df["rh_lag24h"]   = df.groupby("adm_cd2")["rh_pct"].shift(LAG_24H)
    df["hi_lag24h"]   = df.groupby("adm_cd2")["hi_c"].shift(LAG_24H)
    df["wbgt_lag24h"] = df.groupby("adm_cd2")["wbgt_c"].shift(LAG_24H)

    # 변화량
    df["temp_diff_1h"]  = df["temp_c"] - df["temp_lag1h"]
    df["rh_diff_1h"]    = df["rh_pct"] - df["rh_lag1h"]
    df["hi_diff_1h"]    = df["hi_c"]   - df["hi_lag1h"]
    df["wbgt_diff_1h"]  = df["wbgt_c"] - df["wbgt_lag1h"]

    # 이동평균(동별)
    for w in MA_WINDOWS:
        df[f"temp_ma{w}h"]  = df.groupby("adm_cd2")["temp_c"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"rh_ma{w}h"]    = df.groupby("adm_cd2")["rh_pct"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"hi_ma{w}h"]    = df.groupby("adm_cd2")["hi_c"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"wbgt_ma{w}h"]  = df.groupby("adm_cd2")["wbgt_c"].transform(lambda s: s.rolling(w, min_periods=1).mean())

    # 임계 이상 시간수(최근 THRESH_WINDOW시간)
    def rolling_count(series: pd.Series, threshold: float):
        return series.rolling(THRESH_WINDOW, min_periods=1).apply(lambda x: float(np.sum(x >= threshold)), raw=True)

    df["hours_hi_ge30_last6h"]    = df.groupby("adm_cd2")["hi_c"].transform(lambda s: rolling_count(s, 30.0))
    df["hours_wbgt_ge28_last6h"]  = df.groupby("adm_cd2")["wbgt_c"].transform(lambda s: rolling_count(s, 28.0))
    df["hours_wbgt_ge31_last6h"]  = df.groupby("adm_cd2")["wbgt_c"].transform(lambda s: rolling_count(s, 31.0))

    return df

# -------------------------
# Join static attributes
# -------------------------
def join_static(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if not STATIC_PATH.exists():
        print(f"[WARN] static not found: {STATIC_PATH} (skip join)")
        return df
    st = pd.read_csv(STATIC_PATH)
    if "adm_cd2" not in st.columns:
        print("[WARN] static.csv must include 'adm_cd2'; skip join.")
        return df
    return df.merge(st, on="adm_cd2", how="left")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 1) 소스 로드
    sources = []
    for p in SRC_FILES:
        df = _safe_read(p)
        if not df.empty:
            if "ncst_by_dong_latest.csv" in str(p):
                df.source_name = "ncst"
            elif "ufc_by_dong_latest.csv" in str(p):
                df.source_name = "ufc"
            elif "vfc_by_dong_latest.csv" in str(p):
                df.source_name = "vfc"
            sources.append(df)

    if not sources:
        raise SystemExit("No weather source CSVs found. Run fetch_weather.py first.")

    merged = _merge_sources(sources)
    print(f"[INFO] merged rows: {len(merged):,}")

    # 2) 시간 정규화 & 기초 파생(HI/WBGT)
    hourly_base = make_hourly_base(merged)
    print(f"[INFO] base hourly rows: {len(hourly_base):,}")

    # 3) 시간축 완성 + 결측 보정(ffill 제한)
    hourly_full = _fill_and_complete_hourly(hourly_base, ffill_hours=FFILL_HOURS)
    print(f"[INFO] completed hourly rows: {len(hourly_full):,}")

    # 4) 시계열 피처 생성
    hourly_feat = add_hourly_features(hourly_full)
    print(f"[INFO] hourly feature rows: {len(hourly_feat):,}")

    # 5) 정적 데이터 조인
    hourly_feat = join_static(hourly_feat)

    # 6) 저장 (전체 + 실시간 분리)
    now_floor = datetime.now(KST).replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
    hourly_feat["is_future"] = hourly_feat["dt_hour"] > now_floor
    hourly_realtime = hourly_feat[hourly_feat["dt_hour"] <= now_floor].copy()

    ts = _ts()
    out_ts_all = OUT_DIR / f"hourly_features_{ts}.csv"
    out_latest_all = OUT_DIR / "hourly_features_latest.csv"
    out_ts_rt = OUT_DIR / f"hourly_features_realtime_{ts}.csv"
    out_latest_rt = OUT_DIR / "hourly_features_realtime_latest.csv"

    hourly_feat.to_csv(out_ts_all, index=False)
    hourly_feat.to_csv(out_latest_all, index=False)
    hourly_realtime.to_csv(out_ts_rt, index=False)
    hourly_realtime.to_csv(out_latest_rt, index=False)

    print(f"[SAVE] {out_ts_all.name} / hourly_features_latest.csv (ALL incl. forecast)")
    print(f"[SAVE] {out_ts_rt.name} / hourly_features_realtime_latest.csv (<= now)")
