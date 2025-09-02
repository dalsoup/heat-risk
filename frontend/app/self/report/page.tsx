"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { postSelfReport } from "@/lib/api";

const OPTIONS = ["평소보다 높은 체온","두통","어지럼증","메스꺼움","지나친 땀","구역감","갑작스러운 피로감","시야 혼탁","의식 저하","구토","극심한 두통"];

export default function SelfReportPage() {
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const toggle = (s: string) =>
    setSelected((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));

  const submit = async () => {
    try {
      setLoading(true);
      const payload = await postSelfReport(selected);
      const q = encodeURIComponent(JSON.stringify(payload));
      router.push(`/self/result?data=${q}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-3 px-4 pb-28 space-y-4">
      <div className="text-2xl font-extrabold text-rose-500 mt-4">온열질환 자가신고</div>
      <div className="mt-6 text-xl font-semibold leading-relaxed">현재 느껴지는 증상을 모두 체크해주세요.</div>
      <div className="mt-6 rounded-2xl bg-gray-100 p-4 space-y-4">
        {OPTIONS.map((label) => (
          <label key={label} className="flex items-center gap-4 text-xl font-extrabold">
            <input type="checkbox" className="size-6 accent-emerald-600 rounded"
              checked={selected.includes(label)} onChange={() => toggle(label)} />
            {label}
          </label>
        ))}
      </div>
      <button onClick={submit} disabled={loading}
        className="mt-8 w-full h-16 rounded-2xl bg-emerald-800 text-white text-3xl font-extrabold disabled:opacity-60">
        {loading ? "제출 중…" : "제 출"}
      </button>
    </div>
  );
}
