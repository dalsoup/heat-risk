// frontend/lib/api.ts

// -------- Types --------
export type HealthSummary = {
  location: { adm_cd2: string; dong_name: string };
  heat: {
    headline: string;
    risk_score: number;
    score_breakdown: { water: number; cool_rest: number; ambient_c: number };
  };
  hydration: { current_ml: number; goal_ml: number };
  shelters: { nearby_count: number; radius_m: number };
  vitals: { hr_bpm: number; hr_range: [number, number]; spo2_pct: number; spo2_range: [number, number] };
  sleep: { last_night_hours: number };
  updated_at: string;
};

export type SelfReportResult = {
  level: "mild" | "severe";
  headline: string;
  recommendations: string[];
  actions: { shelter: boolean; hospital: boolean; call119: boolean };
};

// -------- Helpers --------
async function readErr(r: Response): Promise<string> {
  let body = "";
  try {
    const text = await r.text();
    try {
      const j = JSON.parse(text);
      if (j?.detail) {
        body = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } else {
        body = text;
      }
    } catch {
      body = text;
    }
  } catch {
    /* ignore */
  }
  return `self-report ${r.status}${body ? `: ${body}` : ""}`;
}

function getLocLabel(): string {
  try {
    return localStorage.getItem("location.label") || "이촌동, 서울";
  } catch {
    return "이촌동, 서울";
  }
}

// -------- API functions --------
export async function getHealthSummary(): Promise<HealthSummary> {
  const r = await fetch("/api/health/summary", { cache: "no-store" });
  if (!r.ok) throw new Error(`summary ${r.status}`);
  return r.json();
}

export async function addHydration(amount_ml: number): Promise<{ current_ml: number; goal_ml: number }> {
  const r = await fetch("/api/health/hydration/add", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ amount_ml }),
  });
  if (!r.ok) throw new Error(`hydration ${r.status}`);
  return r.json();
}

/**
 * 자가신고 전송
 * - 기본 스키마(payloadA) → 422면 payloadB → 422면 payloadC 순서로 재시도
 * - 서버는 보통 추가 필드를 무시하므로 ts/location을 동봉(안전)
 * - 에러 메시지는 서버 응답(detail)을 최대한 보여줌
 */
export async function postSelfReport(symptoms: string[]): Promise<SelfReportResult> {
  if (!Array.isArray(symptoms) || symptoms.length === 0) {
    throw new Error("최소 1개 이상의 증상을 선택해 주세요.");
  }

  const ts = new Date().toISOString();
  const location = getLocLabel();

  // A) 가장 보편적: { symptoms: string[], ts, location }
  const payloadA = { symptoms, ts, location };

  // B) 라벨 대신 코드(symptoms_code)만 받는 경우를 대비 (라벨 → 코드 매핑)
  const CODE_MAP: Record<string, string> = {
    "평소보다 높은 체온": "fever_high",
    "두통": "headache",
    "어지럼증": "dizziness",
    "메스꺼움": "nausea",
    "지나친 땀": "sweating_excess",
    "구역감": "retching",
    "갑작스러운 피로감": "sudden_fatigue",
    "시야 혼탁": "vision_blur",
    "의식 저하": "consciousness_drop",
    "구토": "vomit",
    "극심한 두통": "headache_severe",
  };
  const payloadB = {
    symptoms: symptoms.map((s) => CODE_MAP[s] ?? s),
    ts,
    location,
  };

  // C) 드물게 {items: [...]} 형태를 쓰는 서버
  const payloadC = { items: symptoms, ts, location };

  const url = "/api/health/self";

  // try A
  let r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadA),
  });
  if (r.ok) return r.json();
  if (r.status !== 422) throw new Error(await readErr(r));

  // try B
  r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadB),
  });
  if (r.ok) return r.json();
  if (r.status !== 422) throw new Error(await readErr(r));

  // try C
  r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadC),
  });
  if (r.ok) return r.json();

  // 모두 실패 → 마지막 응답을 에러로
  throw new Error(await readErr(r));
}
