// app/self/report/page.tsx
"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { ClipboardCheck } from "lucide-react";
import { postSelfReport } from "@/lib/api";

const OPTIONS = [
  "평소보다 높은 체온",
  "두통",
  "어지럼증",
  "메스꺼움",
  "지나친 땀",
  "구역감",
  "갑작스러운 피로감",
  "시야 혼탁",
  "의식 저하",
  "구토",
  "극심한 두통",
];

export default function SelfReportPage() {
  const router = useRouter();
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const canSubmit = selected.length > 0 && !loading;

  const toggle = (s: string) =>
    setSelected((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]
    );

  const submit = async () => {
    if (!canSubmit) {
      setToast("최소 1개 이상의 증상을 선택해 주세요.");
      setTimeout(() => setToast(null), 2200);
      return;
    }
    try {
      setLoading(true);
      const payload = await postSelfReport(selected);
      const q = encodeURIComponent(JSON.stringify(payload));
      router.push(`/self/result?data=${q}`);
    } catch (e: any) {
      setToast(e?.message || "제출 중 오류가 발생했습니다.");
      setTimeout(() => setToast(null), 2800);
    } finally {
      setLoading(false);
    }
  };

  return (
    // 이 페이지는 흰색 배경(상단바도 HeaderBar가 흰색으로 전환)
    <div className="-mx-4 bg-white px-4 pb-28">
      {/* 제목 */}
      <div className="mt-6 flex items-center gap-2">
        <ClipboardCheck className="size-7 text-rose-500" strokeWidth={2.5} />
        <h1 className="text-[22px] font-black text-rose-500">온열질환 자가신고</h1>
      </div>

      {/* 설명 */}
      <p className="mt-6 text-[18px] font-extrabold leading-[1.6] text-gray-900">
        현재 느껴지는 증상을 모두 체크해주세요.
      </p>

      {/* 체크리스트 카드 */}
      <div className="mt-5 rounded-[20px] bg-[#EFF1F3] p-4 space-y-4">
        {OPTIONS.map((label) => (
          <label key={label} className="flex items-center gap-4">
            <input
              type="checkbox"
              className="size-6 rounded accent-emerald-600"
              checked={selectedSet.has(label)}
              onChange={() => toggle(label)}
            />
            <span className="text-[18px] font-extrabold text-gray-900">
              {label}
            </span>
          </label>
        ))}
      </div>

      {/* 제출 버튼: 선택 없으면 비활성화 */}
      <button
        onClick={submit}
        disabled={!canSubmit}
        className="mt-8 w-full h-[56px] rounded-2xl bg-emerald-800 text-white text-[22px] font-extrabold
                   disabled:opacity-40 disabled:cursor-not-allowed active:scale-[0.99] transition"
      >
        {loading ? "제출 중…" : "제 출"}
      </button>

      {/* 가벼운 토스트 */}
      {toast && (
        <div
          role="status"
          className="fixed left-1/2 -translate-x-1/2 bottom-24 z-30 px-4 py-2 rounded-full bg-black/80 text-white text-sm"
        >
          {toast}
        </div>
      )}
    </div>
  );
}
