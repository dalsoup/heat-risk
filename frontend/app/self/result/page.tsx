"use client";

import { useMemo } from "react";
import { useSearchParams, useRouter } from "next/navigation";

// ----------------------
// Types & helpers
// ----------------------
type Result = {
  level: "mild" | "severe";
  headline: string;
  recommendations: string[];
  actions: { shelter: boolean; hospital: boolean; call119: boolean };
};

function parseResultParam(raw: string | null): Result | null {
  if (!raw) return null;
  try { const s = decodeURIComponent(atob(raw)); return JSON.parse(s); } catch {}
  try { return JSON.parse(decodeURIComponent(raw)); } catch {}
  try { return JSON.parse(raw); } catch {}
  return null;
}

// ----------------------
// Page
// ----------------------
export default function ResultPage() {
  const sp = useSearchParams();
  const router = useRouter();
  const res = useMemo(() => parseResultParam(sp.get("data")), [sp]);

  if (!res) {
    return (
      <div className="min-h-[100dvh] bg-[#E6F2F3] p-6">
        <div className="text-sm text-gray-600">결과 정보가 없습니다. 자가신고부터 진행해주세요.</div>
        <button
          onClick={() => router.push("/self/report")}
          className="mt-4 inline-flex h-11 items-center rounded-xl bg-emerald-700 px-4 text-white font-bold"
        >
          자가신고 하러가기
        </button>
      </div>
    );
  }

  const isMild = res.level === "mild";
  const accent = isMild ? "text-blue-800" : "text-red-700";
  const containerBg = isMild ? "bg-[#E6F2F3]" : "bg-[#FDE7EA]";

  return (
    <div className={`min-h-[100dvh] ${containerBg} flex flex-col pb-28`}>
      {/* 상단 여백 (전역 위치 헤더와 겹치지 않게) */}
      <div className="h-3" />

      {/* 타이틀 */}
      <div className="px-5 mt-2">
        <div className="text-[22px] font-extrabold text-gray-900">온열질환 자가신고 결과</div>
      </div>

      {/* 카드 */}
      <div className="px-5 mt-4">
        <div className="rounded-[20px] bg-white shadow-sm p-6">
          {/* 서브카피 */}
          <div className="text-center text-[18px] text-gray-700">현재 상태는</div>

          {/* 아이콘 + 헤드라인 */}
          <div className="mt-3 flex flex-col items-center">
            <StatusIcon severity={isMild ? "mild" : "severe"} />
            <div className={`mt-4 text-center text-[36px] leading-tight font-extrabold ${accent}`}>
              {res.headline}
            </div>

            {/* 안내 문구 */}
            {isMild ? (
              <div className="mt-3 text-center text-gray-600 whitespace-pre-line">
                {"즉각적인 심각 위험은 아니지만,\n아래 권장 사항을 따라주시고\n건강 상태를 면밀히 관찰해주세요."}
              </div>
            ) : (
              <div className="mt-3 text-center text-gray-600">
                <span>즉시 대응이 필요합니다.</span>
                <br />
                <span>아래 조치를 지체 없이 수행하세요.</span>
              </div>
            )}

            {/* 면책 문구(줄바꿈 포함, 경미일 때 노출) */}
            {isMild && (
              <div className="mt-2 text-center text-[11px] text-gray-400 whitespace-pre-line">
                {"* 본 결과는 진단이 아닌 추정이며,\n정확한 진료는 가까운 병의원 내원 후 받으시기 바랍니다."}
              </div>
            )}
          </div>

          {/* 구분선 */}
          <div className="my-6 h-px bg-gray-200" />

          {/* 권장 행동 (자간/줄간격 타이트) */}
          <div>
            <div className="text-[18px] font-extrabold mb-2">권장 행동</div>
            <ul className="list-disc pl-5 text-[15px] tracking-[-0.015em] leading-[1.25] space-y-1.5">
              {res.recommendations.map((t, i) => (
                <li key={i} className="text-gray-800">{t}</li>
              ))}
            </ul>
          </div>

          {/* 액션 버튼 */}
          {isMild ? (
            // 경미: 병의원/쉼터 두 버튼 나란히(항상 표시)
            <div className="mt-6 grid grid-cols-2 gap-3">
              <button
                className="h-14 rounded-xl bg-[#0B2A6F] text-white font-extrabold"
                onClick={() => router.push("/hospitals")}
              >
                병의원 찾기
              </button>
              <button
                className="h-14 rounded-xl bg-emerald-800 text-white font-extrabold"
                onClick={() => router.push("/shelters")}
              >
                인근 쉼터 찾기
              </button>
            </div>
          ) : (
            // 심각: 119/병의원 나란히
            <div className="mt-6 grid grid-cols-2 gap-3">
              <a
                href="tel:119"
                className="h-14 rounded-xl bg-[#C1121F] text-white text-[18px] font-extrabold flex items-center justify-center"
              >
                119 신고
              </a>
              <button
                className="h-14 rounded-xl bg-[#0B2A6F] text-white font-extrabold"
                onClick={() => router.push("/hospitals")}
              >
                병의원 찾기
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 하단 탭바 */}
      <nav className="fixed inset-x-0 bottom-0 z-10 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/70 border-t border-gray-200">
        <div className="mx-auto max-w-md px-6 py-3 grid grid-cols-4 gap-4 text-[12px]">
          <Tab icon="home"    label="홈"        href="/"            active={false} />
          <Tab icon="cart"    label="포인트"    href="/points"      active={false} />
          <Tab icon="check"   label="자가진단"  href="/self/report" active />
          <Tab icon="user"    label="마이페이지" href="/mypage"      active={false} />
        </div>
      </nav>
    </div>
  );
}

// ----------------------
// Status Icon (mild ↔ severe)
// ----------------------
function StatusIcon({ severity }: { severity: "mild" | "severe" }) {
  if (severity === "severe") {
    // 🚨 빨간 경광등
    return (
      <svg
        width="120"
        height="120"
        viewBox="0 0 120 120"
        role="img"
        aria-label="심각한 온열질환 경고"
        className="drop-shadow-md"
      >
        <defs>
          <radialGradient id="dome" cx="50%" cy="35%" r="65%">
            <stop offset="0%" stopColor="#FF6B6B" />
            <stop offset="60%" stopColor="#E03131" />
            <stop offset="100%" stopColor="#B51D1D" />
          </radialGradient>
          <radialGradient id="shine" cx="50%" cy="40%" r="35%">
            <stop offset="0%" stopColor="white" stopOpacity="0.95" />
            <stop offset="60%" stopColor="white" stopOpacity="0.25" />
            <stop offset="100%" stopColor="white" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="base" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2B2B2B" />
            <stop offset="100%" stopColor="#0F0F10" />
          </linearGradient>
          <radialGradient id="glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#FF9AA2" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#FF9AA2" stopOpacity="0" />
          </radialGradient>
          <filter id="blur" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" />
          </filter>
        </defs>

        <ellipse cx="60" cy="104" rx="34" ry="6" fill="url(#glow)" filter="url(#blur)" />
        <ellipse cx="60" cy="96" rx="36" ry="10" fill="url(#base)" />
        <ellipse cx="60" cy="92" rx="38" ry="8" fill="#1F1F20" />
        <path d="M28 86c0-26 12-46 32-46s32 20 32 46H28z" fill="url(#dome)" />
        <path d="M28 86c0-26 12-46 32-46s32 20 32 46" fill="none" stroke="#8E1A1A" strokeWidth="2" opacity="0.5" />
        <circle cx="60" cy="58" r="18" fill="url(#shine)" />
      </svg>
    );
  }

  // 💧 경미: 파란 물방울
  return (
    <svg width="80" height="80" viewBox="0 0 64 64" role="img" aria-label="경미한 온열질환">
      <defs>
        <linearGradient id="mildGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#5FA8FF" />
          <stop offset="1" stopColor="#0B5EE8" />
        </linearGradient>
      </defs>
      <path d="M24 8c7 10 12 16 12 24a12 12 0 11-24 0c0-8 5-14 12-24z" fill="url(#mildGrad)"/>
      <path d="M44 18c4 6 7 10 7 15a7 7 0 11-14 0c0-5 3-9 7-15z" fill="url(#mildGrad)" opacity="0.85"/>
      <path d="M50 34c2 3 4 5 4 7a4 4 0 11-8 0c0-2 2-4 4-7z" fill="url(#mildGrad)" opacity="0.7"/>
    </svg>
  );
}

// ----------------------
// Bottom Tab (inline)
// ----------------------
function Tab({
  icon, label, href, active,
}: { icon: "home"|"cart"|"check"|"user"; label: string; href: string; active?: boolean }) {
  const router = useRouter();
  const base = "flex flex-col items-center justify-center gap-1";
  const color = active ? "text-emerald-700" : "text-gray-500";
  return (
    <button onClick={() => router.push(href)} className={`${base} ${color}`}>
      <Icon name={icon} />
      <span className="font-semibold">{label}</span>
    </button>
  );
}

function Icon({ name }: { name: "home"|"cart"|"check"|"user" }) {
  if (name === "home") {
    return (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
        <path d="M3 10.5l9-7 9 7V20a2 2 0 0 1-2 2h-4v-6H9v6H5a2 2 0 0 1-2-2v-9.5z" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
  }
  if (name === "cart") {
    return (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
        <circle cx="9" cy="20" r="1.5" fill="currentColor"/><circle cx="18" cy="20" r="1.5" fill="currentColor"/>
        <path d="M3 4h3l2.2 10.2A2 2 0 0 0 10.1 16h6.9a2 2 0 0 0 2-1.6L21 7H6" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
  }
  if (name === "check") {
    return (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
        <rect x="3" y="3" width="18" height="18" rx="4" stroke="currentColor" strokeWidth="1.7"/>
        <path d="M8 12l3 3 5-6" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
  }
  // user
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="8" r="4" stroke="currentColor" strokeWidth="1.7"/>
      <path d="M4 20c1.7-3.2 4.7-5 8-5s6.3 1.8 8 5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/>
    </svg>
  );
}
