# fetch_weather.py — full replacement (ALL DONGS)
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from dfs_grid import latlon_to_grid
from kma_client import (
    get_ultra_nowcast,
    get_ultra_forecast,
    get_vilage_forecast,
    flatten_items,
)

# -------------------------
# Config
# -------------------------
load_dotenv(override=True)
KST = timezone(timedelta(hours=9))

RAW_DIR = Path("data/weather_raw")
PARSED_DIR = Path("data/weather_parsed")
ADM_PATH = Path("data/admin_dong.csv")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# polite delay between KMA calls (seconds) – adjust if needed
SLEEP_BETWEEN_CALLS = float(os.getenv("KMA_CALL_SLEEP", "0.2"))

def _ts() -> str:
    return datetime.now(KST).strftime("%Y%m%d%H%M%S")

def _save_json_with_latest(prefix: str, nx: int, ny: int, payload: dict):
 
    ts = _ts()
    p_ts = RAW_DIR / f"{prefix}_{nx}_{ny}_{ts}.json"
    p_latest = RAW_DIR / f"{prefix}_{nx}_{ny}_latest.json"
    with p_ts.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with p_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] raw -> {p_ts.name} (and {p_latest.name})")

def _save_df_with_latest(prefix: str, nx: int, ny: int, df: pd.DataFrame):
 
    ts = _ts()
    p_ts = PARSED_DIR / f"{prefix}_{nx}_{ny}_{ts}.csv"
    p_latest = PARSED_DIR / f"{prefix}_{nx}_{ny}_latest.csv"
    df.to_csv(p_ts, index=False)
    df.to_csv(p_latest, index=False)
    print(f"[SAVE] parsed -> {p_ts.name} (and {p_latest.name})")

def _save_df_global(prefix: str, df: pd.DataFrame):

    ts = _ts()
    p_ts = PARSED_DIR / f"{prefix}_by_dong_{ts}.csv"
    p_latest = PARSED_DIR / f"{prefix}_by_dong_latest.csv"
    df.to_csv(p_ts, index=False)
    df.to_csv(p_latest, index=False)
    print(f"[SAVE] global parsed -> {p_ts.name} (and {p_latest.name})")

# -------------------------
# Parsers (timezone-safe)
# -------------------------
def _parse_ncst_to_df(items: list) -> pd.DataFrame:

    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    base_date = str(df["baseDate"].iloc[0])
    base_time = str(df["baseTime"].iloc[0])
    ts = pd.to_datetime(base_date + base_time, format="%Y%m%d%H%M").tz_localize(KST)

    wide = (
        df[["category", "obsrValue"]]
        .dropna()
        .assign(obsrValue=lambda x: pd.to_numeric(x["obsrValue"], errors="coerce"))
        .pivot_table(index=None, columns="category", values="obsrValue", aggfunc="last")
        .reset_index(drop=True)
    )
    wide.insert(0, "dt", ts)
    cols = ["dt"] + sorted([c for c in wide.columns if c != "dt"])
    return wide[cols]

def _parse_fcst_to_df(items: list) -> pd.DataFrame:

    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    if not {"fcstDate", "fcstTime"}.issubset(df.columns):
        return pd.DataFrame()

    df["dt"] = pd.to_datetime(
        df["fcstDate"].astype(str) + df["fcstTime"].astype(str),
        format="%Y%m%d%H%M",
    ).dt.tz_localize(KST)

    wide = (
        df[["dt", "category", "fcstValue"]]
        .dropna()
        .assign(fcstValue=lambda x: pd.to_numeric(x["fcstValue"], errors="coerce"))
        .pivot_table(index="dt", columns="category", values="fcstValue", aggfunc="last")
        .sort_index()
        .reset_index()
    )
    cols = ["dt"] + sorted([c for c in wide.columns if c != "dt"])
    return wide[cols]

# -------------------------
# Admin loader and grid grouping
# -------------------------
def _load_admin() -> pd.DataFrame:
    if not ADM_PATH.exists():
        raise FileNotFoundError(f"Missing file: {ADM_PATH}")
    df = pd.read_csv(ADM_PATH)
    required = {"adm_cd2", "adm_nm", "lat", "lon"}
    if not required.issubset(df.columns):
        raise ValueError(f"admin_dong.csv must contain columns: {sorted(list(required))}")
    return df

def _append_grid_cols(df: pd.DataFrame) -> pd.DataFrame:
    # lat/lon -> nx, ny
    nx_list, ny_list = [], []
    for lat, lon in zip(df["lat"], df["lon"]):
        nx, ny = latlon_to_grid(lat, lon)
        nx_list.append(nx)
        ny_list.append(ny)
    df = df.copy()
    df["nx"] = nx_list
    df["ny"] = ny_list
    return df

# -------------------------
# Per-grid fetch
# -------------------------
def _fetch_one_grid(nx: int, ny: int):

    # 초단기실황
    ncst_json = get_ultra_nowcast(nx, ny)
    _save_json_with_latest("ncst", nx, ny, ncst_json)
    ncst_df = _parse_ncst_to_df(flatten_items(ncst_json))
    _save_df_with_latest("ncst", nx, ny, ncst_df)
    time.sleep(SLEEP_BETWEEN_CALLS)

    # 초단기예보
    ufc_json = get_ultra_forecast(nx, ny)
    _save_json_with_latest("ufc", nx, ny, ufc_json)
    ufc_df = _parse_fcst_to_df(flatten_items(ufc_json))
    _save_df_with_latest("ufc", nx, ny, ufc_df)
    time.sleep(SLEEP_BETWEEN_CALLS)

    # 단기예보
    vfc_json = get_vilage_forecast(nx, ny)
    _save_json_with_latest("vfc", nx, ny, vfc_json)
    vfc_df = _parse_fcst_to_df(flatten_items(vfc_json))
    _save_df_with_latest("vfc", nx, ny, vfc_df)

    return ncst_df, ufc_df, vfc_df

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Admin 전체 로드 & (nx,ny) 변환
    adm_df = _load_admin()
    adm_df = _append_grid_cols(adm_df)

    # (nx,ny) 그리드 단위로 dedupe
    grid_groups = adm_df.groupby(["nx", "ny"])

    print(f"[INFO] total dongs: {len(adm_df):,} | unique grids: {len(grid_groups):,}")

    # 그리드별 호출 결과를 행정동으로 확장하여 결합 저장할 버퍼
    all_ncst_rows = []
    all_ufc_rows = []
    all_vfc_rows = []

    for (nx, ny), group in grid_groups:
        print(f"\n[GRID] nx={nx}, ny={ny} | dongs in this grid: {len(group)}")

        try:
            ncst_df, ufc_df, vfc_df = _fetch_one_grid(nx, ny)
        except Exception as e:
            print(f"[ERROR] grid ({nx},{ny}) fetch failed: {e}")
            continue

        # 그리드 결과를 행정동 단위로 복제(각 행정동 메타를 붙임)
        # ---- 실황 (단일 시점) ----
        if not ncst_df.empty:
            base = ncst_df.copy()
            base["nx"] = nx
            base["ny"] = ny
            # 각 dong 복제
            for _, row in group.iterrows():
                tmp = base.copy()
                tmp["adm_cd2"] = row["adm_cd2"]
                tmp["adm_nm"] = row["adm_nm"]
                all_ncst_rows.append(tmp)

        # ---- 예보(다수 시점) ----
        if not ufc_df.empty:
            base = ufc_df.copy()
            base["nx"] = nx
            base["ny"] = ny
            for _, row in group.iterrows():
                tmp = base.copy()
                tmp["adm_cd2"] = row["adm_cd2"]
                tmp["adm_nm"] = row["adm_nm"]
                all_ufc_rows.append(tmp)

        if not vfc_df.empty:
            base = vfc_df.copy()
            base["nx"] = nx
            base["ny"] = ny
            for _, row in group.iterrows():
                tmp = base.copy()
                tmp["adm_cd2"] = row["adm_cd2"]
                tmp["adm_nm"] = row["adm_nm"]
                all_vfc_rows.append(tmp)

    # ---- 글로벌(by_dong) 저장 ----
    def _concat_or_empty(lst):
        return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()

    g_ncst = _concat_or_empty(all_ncst_rows)
    g_ufc = _concat_or_empty(all_ufc_rows)
    g_vfc = _concat_or_empty(all_vfc_rows)

    # 컬럼 정렬: dt, adm_cd2, adm_nm, nx, ny, (나머지 카테고리)
    def _order_cols(df: pd.DataFrame):
        if df.empty:
            return df
        front = [c for c in ["dt", "adm_cd2", "adm_nm", "nx", "ny"] if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        # dt가 맨 앞, 나머지는 알파벳 정렬
        rest_sorted = sorted(rest)
        return df[front + rest_sorted]

    g_ncst = _order_cols(g_ncst)
    g_ufc = _order_cols(g_ufc)
    g_vfc = _order_cols(g_vfc)

    if not g_ncst.empty:
        _save_df_global("ncst", g_ncst)
    if not g_ufc.empty:
        _save_df_global("ufc", g_ufc)
    if not g_vfc.empty:
        _save_df_global("vfc", g_vfc)

    print("\n[DONE] all grids processed.")
