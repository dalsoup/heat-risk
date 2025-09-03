// app/page.tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import RingScore from "@/components/RingScore";
import SegmentBar from "@/components/SegmentBar";
import { getHealthSummary, addHydration, type HealthSummary } from "@/lib/api";

import {
  Sun,
  AlertTriangle,
  Droplet,
  Umbrella,
  HeartPulse,
  Percent,
  AlarmClock,
} from "lucide-react";

export default function Page() {
  const [d, setD] = useState<HealthSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const j = await getHealthSummary();
        setD(j);
      } catch (e: any) {
        setErr(e?.message ?? "load error");
      }
    })();
  }, []);

  const hydrationPct = useMemo(() => {
    const cur = d?.hydration?.current_ml ?? 0;
    const goal = d?.hydration?.goal_ml ?? 1;
    return Math.min(100, Math.round((cur / goal) * 100));
  }, [d?.hydration?.current_ml, d?.hydration?.goal_ml]);

  if (err) return <div className="p-6 text-sm text-red-600">에러: {err}</div>;
  if (!d) return <div className="p-6 text-sm text-gray-500">불러오는 중…</div>;

  return (
    <div className="mt-3 space-y-4">
      {/* (카드) 폭염 배너 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-full bg-gray-100 flex items-center justify-center text-gray-400">
            <Sun size={18} strokeWidth={1.8} />
          </span>
          <div className="text-lg font-semibold">폭염</div>
        </div>
        <div className="opacity-0">토글</div>
      </section>

      {/* (카드) 실시간 위험점수 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <span className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-amber-600">
              <AlertTriangle size={18} strokeWidth={1.8} />
            </span>
            <div className="text-lg font-semibold">실시간 위험점수</div>
          </div>
          <div className="text-gray-500">{d.heat.headline}</div>
        </div>

        <div className="mt-2 flex items-center justify-between gap-2">
          <div className="text-[13px] text-gray-600 space-y-2">
            <div className="flex justify-between gap-6"><span>물 섭취 기여</span><span>{d.heat.score_breakdown.water}%</span></div>
            <div className="flex justify-between gap-6"><span>쿨링 휴식</span><span>{d.heat.score_breakdown.cool_rest}%</span></div>
            <div className="flex justify-between gap-6"><span>주변 체감</span><span>{d.heat.score_breakdown.ambient_c}℃</span></div>
            <div className="text-gray-400 text-[12px]">새로고침 {new Date(d.updated_at).toLocaleTimeString()}</div>
          </div>

          <div className="shrink-0">
            <RingScore value={d.heat.risk_score} size={150} stroke={16} />
          </div>
        </div>
      </section>

      {/* (카드) 수분 섭취 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-sky-600">
              <Droplet size={18} strokeWidth={1.8} />
            </span>
            <div className="text-xl font-extrabold">
              {d.hydration.current_ml.toLocaleString()}
              <span className="text-gray-300 font-bold"> / {d.hydration.goal_ml.toLocaleString()} ml</span>
            </div>
          </div>
          <button
            className="mt-3 h-10 px-4 rounded-full border border-gray-300 text-gray-900 text-sm"
            onClick={async () => {
              const j = await addHydration(250);
              setD(v => v ? ({ ...v, hydration: { ...v.hydration, current_ml: j.current_ml } }) : v);
            }}
          >
            + 250 ml
          </button>
        </div>

        {/* 물 채워지는 애니메이션 */}
        <WaterTank percent={hydrationPct} />
      </section>

      {/* (카드) 무더위 쉼터 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="flex items-center gap-2">
          <span className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-400">
            <Umbrella size={18} strokeWidth={1.8} />
          </span>
          <div className="text-xl font-extrabold">
            {d.shelters.nearby_count}개 <span className="text-gray-300 font-bold">/ {d.shelters.radius_m}m 이내</span>
          </div>
        </div>
        <button className="mt-3 h-10 px-4 rounded-full border border-gray-300 text-gray-900 text-sm">무더위쉼터 방문 인증하기</button>
      </section>

      {/* (카드) 바이탈 1: 심박수 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-400">
              <HeartPulse size={18} strokeWidth={1.8} />
            </span>
            <div className="text-2xl font-extrabold">{d.vitals.hr_bpm}bpm</div>
          </div>
          <div className="w-1/2">
            <SegmentBar min={d.vitals.hr_range[0]} max={d.vitals.hr_range[1]} range={[80, 95]} color="#ef4444" />
          </div>
        </div>
        <button className="mt-3 h-10 px-4 rounded-full border border-gray-300 text-gray-900 text-sm">심박수 그래프 보기</button>
      </section>

      {/* (카드) 바이탈 2: 산소포화도 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-400">
              <Percent size={18} strokeWidth={1.8} />
            </span>
            <div className="text-2xl font-extrabold">{d.vitals.spo2_pct}%</div>
          </div>
          <div className="w-1/2">
            <SegmentBar min={d.vitals.spo2_range[0]} max={d.vitals.spo2_range[1]} range={[92, 97]} color="#38bdf8" />
          </div>
        </div>
        <button className="mt-3 h-10 px-4 rounded-full border border-gray-300 text-gray-900 text-sm">산소포화도 그래프 보기</button>
      </section>

      {/* (카드) 수면 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="flex items-center gap-2">
          <span className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-400">
            <AlarmClock size={18} strokeWidth={1.8} />
          </span>
          <div className="text-xl font-extrabold">{d.sleep.last_night_hours}시간</div>
        </div>
        <div className="mt-2 w-full h-16 rounded-xl bg-purple-100 flex items-center justify-center text-purple-800 font-semibold">
          2:00 AM -
        </div>
      </section>
    </div>
  );
}

/* ==== Animated Water Tank ==== */
function WaterTank({ percent }: { percent: number }) {
  return (
    <div className="relative w-24 h-24 rounded-2xl bg-gradient-to-b from-sky-50 to-blue-100">
      <div className="absolute inset-2 rounded-xl overflow-hidden bg-white shadow-inner">
        <div
          className="absolute bottom-0 left-0 right-0 transition-[height] duration-700 ease-out"
          style={{ height: `${percent}%` }}
        >
          <div className="absolute inset-0 bg-gradient-to-t from-blue-500 to-blue-400" />
          <svg
            viewBox="0 0 120 20"
            preserveAspectRatio="none"
            className="absolute -bottom-1 left-0 w-[200%] h-8 opacity-70 animate-[wave_4s_linear_infinite]"
          >
            <path
              d="M0 10 Q 10 0 20 10 T 40 10 T 60 10 T 80 10 T 100 10 T 120 10 V20 H0 Z"
              fill="rgba(255,255,255,0.35)"
            />
          </svg>
        </div>
      </div>
      <Droplet size={28} strokeWidth={2} className="absolute right-2 top-2 text-blue-700/80" />
    </div>
  );
}
