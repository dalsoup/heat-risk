# kma_client.py — full replacement (drop-in)
import os, ssl, json
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import certifi

# -------------------------
# ENV & base config
# -------------------------
KST = timezone(timedelta(hours=9))
# .env 값이 기존 환경변수를 덮어쓰도록
load_dotenv(override=True)

# 진단용 HTTP/HTTPS 토글 (운영은 HTTPS)
_USE_HTTP = os.getenv("KMA_USE_HTTP", "false").lower() in ("1", "true", "yes")
_BASE = f"{'http' if _USE_HTTP else 'https'}://apis.data.go.kr/1360000"
_VERIFY = False if _USE_HTTP else certifi.where()

# -------------------------
# Session (TLS1.2 + retries)
# -------------------------
class TLS12HttpAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

def _session() -> requests.Session:
    s = requests.Session()
    if not _USE_HTTP:
        # HTTPS에서만 TLS 어댑터 장착
        s.mount(
            "https://",
            TLS12HttpAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "HEAD", "OPTIONS"],
                )
            ),
        )
    else:
        s.mount(
            "http://",
            HTTPAdapter(
                max_retries=Retry(
                    total=2,
                    backoff_factor=0.2,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "HEAD", "OPTIONS"],
                )
            ),
        )
    s.headers.update({"Accept": "application/json"})
    return s

# -------------------------
# Key normalization
# -------------------------
def _strip_all_whitespace(s: str) -> str:
    # 모든 공백/개행/zero-width 제거
    return "".join(s.split())

def _get_service_key() -> str:
    """
    .env의 KMA_SERVICE_KEY를 '원본(raw)' 형태로 반환.
    - %로 인코딩돼 있으면 전부 언코드해서 원본 복구
    - + / = 를 포함해도 그대로 반환 (requests가 한 번만 인코딩함)
    - 숨은 공백/개행/zero-width 제거
    """
    key = os.getenv("KMA_SERVICE_KEY", "")
    key = _strip_all_whitespace(key)
    if not key:
        raise RuntimeError("KMA_SERVICE_KEY 가 설정되지 않았습니다 (.env 확인).")

    # 여러 번 인코딩된 경우를 모두 풀어 '원본'으로
    cur = key
    for _ in range(5):  # 최대 5회만 시도
        new = unquote(cur)
        if new == cur:
            break
        cur = new

    # 진단 출력: %25(=문자 % 자체)가 남아있는지 확인
    if "%25" in cur or "%2" in cur:
        print("[DEBUG] service key still seems encoded-ish (contains %2..). Check your .env.")

    return cur

# -------------------------
# Time helpers (안전한 base_time + 후보 다중 시도)
# -------------------------
def _ultra_now_candidates():
    """
    초단기실황(getUltraSrtNcst): 관측. 정각(HH00).
    현재시각 - 40분 기준으로 최근 정각부터 과거로 최대 4개 후보 생성.
    """
    t0 = datetime.now(KST) - timedelta(minutes=40)
    base0 = t0.replace(minute=0, second=0, microsecond=0)
    for k in range(0, 4):  # 최근 ~ 3시간 전
        tt = base0 - timedelta(hours=k)
        yield tt.strftime("%Y%m%d"), tt.strftime("%H%M")

def _ultra_fcst_candidates():
    """
    초단기예보(getUltraSrtFcst): 30분 간격(HH00/HH30).
    현재시각 -45분 기준 블록으로 내림 후, 30분 단위로 0, -30, -60, -90 재시도.
    """
    t = datetime.now(KST) - timedelta(minutes=45)
    minute_block = 0 if t.minute < 30 else 30
    base = t.replace(minute=minute_block, second=0, microsecond=0)
    for k in range(0, 4):  # 0, -30, -60, -90
        tt = base - timedelta(minutes=30 * k)
        yield tt.strftime("%Y%m%d"), tt.strftime("%H%M")

def _vilage_fcst_slots():
    return [2, 5, 8, 11, 14, 17, 20, 23]

def _vilage_fcst_candidates():
    """
    단기예보(getVilageFcst): 8회/일 {02,05,08,11,14,17,20,23}
    현재시각 -1시간 기준 가장 가까운 이전 발표시각에서 시작,
    과거 슬롯으로 최대 5회까지 재시도(일 경계 자동 처리).
    """
    t = datetime.now(KST) - timedelta(hours=1)
    slots = _vilage_fcst_slots()
    # 시작 인덱스 찾기
    candidates = []
    # 오늘 기준
    today = t.date()
    # 현재 시각 이하인 가장 큰 슬롯
    start_hour = max([h for h in slots if h <= t.hour], default=None)
    if start_hour is None:
        # 모두 초과면 전날 23시
        start_dt = datetime.combine(today - timedelta(days=1), datetime.min.time()).replace(hour=23, tzinfo=KST)
    else:
        start_dt = datetime.combine(today, datetime.min.time()).replace(hour=start_hour, tzinfo=KST)

    # 후보 5개(현재 포함): 슬롯을 거꾸로 돌며 일 경계 넘기
    dt = start_dt
    for _ in range(5):
        candidates.append((dt.strftime("%Y%m%d"), dt.strftime("%H%M")))
        # 이전 슬롯
        prev_idx = slots.index(dt.hour) - 1 if dt.hour in slots else len(slots) - 1
        if prev_idx < 0:
            prev_idx = len(slots) - 1
        prev_hour = slots[prev_idx]
        if prev_hour > dt.hour:
            # 전날로 넘어감
            dt = (dt - timedelta(days=1)).replace(hour=prev_hour)
        else:
            dt = dt.replace(hour=prev_hour)
    return candidates

# -------------------------
# Parser helpers
# -------------------------
def _parse_kma_json_or_raise(r: requests.Response) -> dict:
    """
    공통 요청 후 KMA JSON 파싱 + resultCode 검증.
    XML/HTML/텍스트 응답일 경우 적절한 예외를 발생.
    """
    ctype = r.headers.get("Content-Type", "")
    print("[HTTP]", r.status_code, ctype, "| URL:", r.url)
    r.raise_for_status()

    # JSON 시도
    try:
        j = r.json()
        # KMA 공통 헤더 검사
        header = (
            j.get("response", {})
             .get("header", {})
        )
        result_code = header.get("resultCode")
        result_msg = header.get("resultMsg")
        if result_code and result_code != "00":
            # 상세 바디/메시지 힌트 제공
            raise RuntimeError(f"KMA error resultCode={result_code}, msg={result_msg}")
        return j
    except ValueError:
        # JSON 아님 → XML/텍스트 핸들
        body = r.text.strip()
        if body.startswith("<"):
            try:
                root = ET.fromstring(body)
                code = (
                    root.findtext(".//resultCode")
                    or root.findtext(".//returnReasonCode")
                    or root.findtext(".//code")
                )
                msg = (
                    root.findtext(".//resultMsg")
                    or root.findtext(".//returnAuthMsg")
                    or root.findtext(".//message")
                )
                print("[XML-ERROR]", code, msg)
                raise RuntimeError(f"KMA XML error: code={code} msg={msg}")
            except ET.ParseError:
                print("[BODY-HEAD]", body[:500])
                raise RuntimeError("Non-JSON/XML response (parse error)")
        print("[BODY-HEAD]", body[:500])
        raise RuntimeError("Non-JSON response")

def flatten_items(resp_json):
    """KMA JSON -> items list"""
    body = resp_json.get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])
    return items if isinstance(items, list) else []

def _request_json(url: str, params: dict, timeout=15) -> dict:
    s = _session()
    r = s.get(url, params=params, timeout=timeout, verify=_VERIFY)
    return _parse_kma_json_or_raise(r)

# -------------------------
# Core retry wrapper
# -------------------------
def _try_candidates(build_url_and_params, candidates, timeout=15):
    """
    candidates: iterable of (base_date, base_time)
    build_url_and_params: fn(base_date, base_time) -> (url, params)
    """
    last_err = None
    for i, (bd, bt) in enumerate(candidates, 1):
        url, params = build_url_and_params(bd, bt)
        print(f"[TRY {i}] base_date={bd} base_time={bt}")
        try:
            js = _request_json(url, params, timeout=timeout)
            # items 비었으면 다음 후보 시도 (단, 일부 엔드포인트는 header만 있고 body 없을 수 있어 items 기준)
            items = flatten_items(js)
            if items:
                print(f"[TRY {i}] success: items={len(items)}")
                return js
            else:
                # 그래도 성공으로 볼지 여부는 상황에 따라 다름.
                # 여기선 "실데이터 없음"을 강건하게 처리하기 위해 다음 후보도 시도.
                print(f"[TRY {i}] empty items; trying next candidate...")
                last_err = RuntimeError("Empty items")
        except Exception as e:
            print(f"[TRY {i}] failed: {e}")
            last_err = e
    # 전부 실패/비어있음 → 마지막 에러를 던짐
    raise last_err if last_err else RuntimeError("All candidates failed")

# -------------------------
# Endpoints
# -------------------------
def get_ultra_nowcast(nx: int, ny: int):
    """
    초단기실황(관측)
    - 정각(HH00) 후보를 최근~과거 3개 더 시도하여 공백 타이밍 보강
    """
    service_key = _get_service_key()
    url_base = f"{_BASE}/VilageFcstInfoService_2.0/getUltraSrtNcst"

    def builder(bd: str, bt: str):
        params = {
            "serviceKey": service_key,
            "numOfRows": "100",
            "pageNo": "1",
            "dataType": "JSON",
            "base_date": bd,
            "base_time": bt,
            "nx": str(nx),
            "ny": str(ny),
        }
        return url_base, params

    return _try_candidates(builder, _ultra_now_candidates(), timeout=15)

def get_ultra_forecast(nx: int, ny: int):
    """
    초단기예보(1~6시간)
    - 30분 블록(HH00/HH30)에서 0, -30, -60, -90분 후보 재시도
    """
    service_key = _get_service_key()
    url_base = f"{_BASE}/VilageFcstInfoService_2.0/getUltraSrtFcst"

    def builder(bd: str, bt: str):
        params = {
            "serviceKey": service_key,
            "numOfRows": "1000",
            "pageNo": "1",
            "dataType": "JSON",
            "base_date": bd,
            "base_time": bt,
            "nx": str(nx),
            "ny": str(ny),
        }
        return url_base, params

    return _try_candidates(builder, _ultra_fcst_candidates(), timeout=15)

def get_vilage_forecast(nx: int, ny: int):
    """
    단기예보(3일)
    - 발표시각 슬롯(02,05,08,11,14,17,20,23)에서 최대 5회 과거 슬롯 재시도
    """
    service_key = _get_service_key()
    url_base = f"{_BASE}/VilageFcstInfoService_2.0/getVilageFcst"

    def builder(bd: str, bt: str):
        params = {
            "serviceKey": service_key,
            "numOfRows": "2000",
            "pageNo": "1",
            "dataType": "JSON",
            "base_date": bd,
            "base_time": bt,
            "nx": str(nx),
            "ny": str(ny),
        }
        return url_base, params

    return _try_candidates(builder, _vilage_fcst_candidates(), timeout=20)
