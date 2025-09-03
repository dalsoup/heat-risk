"use client";

import { usePathname, useSearchParams } from "next/navigation";
import HeaderBar from "@/components/HeaderBar";
import TabBar from "@/components/TabBar";

/** ?data 파라미터에서 level 추출 (base64/URI/json 모두 대응) */
function parseLevelFromParams(sp: ReturnType<typeof useSearchParams>): "mild" | "severe" | null {
  const raw = sp.get("data");
  if (!raw) return null;
  const tryParse = (s: string) => {
    try { return JSON.parse(s) as { level?: string } } catch { return null; }
  };
  // 1) base64 → decodeURIComponent → JSON
  try {
    const s1 = atob(raw);
    try { const s2 = decodeURIComponent(s1); const j = JSON.parse(s2); return (j.level === "severe" ? "severe" : j.level === "mild" ? "mild" : null); } catch {}
    const j = tryParse(s1); if (j?.level === "severe" || j?.level === "mild") return j.level as any;
  } catch {}
  // 2) uri-encoded json
  try { const j = JSON.parse(decodeURIComponent(raw)); if (j?.level === "severe" || j?.level === "mild") return j.level as any; } catch {}
  // 3) plain json
  try { const j = JSON.parse(raw); if (j?.level === "severe" || j?.level === "mild") return j.level as any; } catch {}
  return null;
}

/** 경로/파라미터 기반 배경 테마 */
function useTheme() {
  const pathname = usePathname() || "/";
  const sp = useSearchParams();

  // self/report → 흰색
  if (pathname === "/self/report" || pathname.startsWith("/self/report/")) {
    return { wrapBg: "bg-white", headerBg: "bg-white" };
  }

  // self/result → level에 따라 하늘색/핑크
  if (pathname === "/self/result" || pathname.startsWith("/self/result/")) {
    const level = parseLevelFromParams(sp);
    if (level === "severe") {
      return { wrapBg: "bg-[#FDE7EA]", headerBg: "bg-[#FDE7EA]" }; 
    }
    return { wrapBg: "bg-[#E6F2F3]", headerBg: "bg-[#E6F2F3]" };   
  }

  return { wrapBg: "bg-[#f2f3f5]", headerBg: "bg-[#f2f3f5]" };
}

export default function HeaderWrapper({ children }: { children: React.ReactNode }) {
  const { wrapBg, headerBg } = useTheme();

  return (
    <div className={`mx-auto w-full max-w-[420px] min-h-dvh flex flex-col ${wrapBg}`}>
      {/* 상태바 여백 */}
      <div className="h-5" />

      {/* 상단바도 동일 배경으로 */}
      <HeaderBar forceBg={headerBg} />

      {/* 본문 */}
      <main className="flex-1 px-4 pb-28">{children}</main>

      {/* 탭바 */}
      <nav className="fixed inset-x-0 bottom-0 z-50">
        <div className="mx-auto w-full max-w-[420px] px-4 pb-[env(safe-area-inset-bottom)]">
          <TabBar />
        </div>
      </nav>
    </div>
  );
}
