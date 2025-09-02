// frontend/lib/api.ts
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

export type SelfReportResult = {
  level: "mild" | "severe";
  headline: string;
  recommendations: string[];
  actions: { shelter: boolean; hospital: boolean; call119: boolean };
};

export async function postSelfReport(symptoms: string[]): Promise<SelfReportResult> {
  const r = await fetch("/api/health/self", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms }),
  });
  if (!r.ok) throw new Error(`self-report ${r.status}`);
  return r.json();
}
