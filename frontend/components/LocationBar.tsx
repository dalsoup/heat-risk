"use client";

import { useEffect, useState } from "react";
import { MapPin } from "lucide-react";

export default function LocationBar({
  defaultText = "청운효자동, 서울",
  bg = "bg-[#f2f3f5]", // 기본 회색
  showToggle = true,
}: {
  defaultText?: string;
  bg?: string;        // "bg-white" | "bg-[#f2f3f5]" 등
  showToggle?: boolean;
}) {
  const [label, setLabel] = useState(defaultText);
  const [shared, setShared] = useState(false);

  useEffect(() => {
    // ?loc= → localStorage → 기본값
    try {
      const url = new URL(window.location.href);
      const q = url.searchParams.get("loc");
      if (q && q.trim()) {
        setLabel(q.trim());
        localStorage.setItem("location.label", q.trim());
        return;
      }
    } catch {}
    try {
      const saved = localStorage.getItem("location.label");
      if (saved && saved.trim()) setLabel(saved.trim());
    } catch {}
  }, [defaultText]);

  return (
    <div className={`${bg} px-5 pt-3 pb-2`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MapPin className="size-5 text-gray-800" strokeWidth={2.5} />
          <span className="text-[17px] font-semibold text-gray-900">{label}</span>
        </div>

        {showToggle && (
          <label className="relative inline-flex items-center">
            <input
              type="checkbox"
              className="peer sr-only"
              checked={shared}
              onChange={(e) => setShared(e.target.checked)}
              aria-label="데이터 공유"
            />
            {/* iOS 느낌 토글 (연회색) */}
            <div className="w-[52px] h-[30px] rounded-full bg-gray-300 transition-colors peer-checked:bg-gray-300/70 relative">
              <div className="absolute top-0.5 left-0.5 size-[26px] rounded-full bg-white shadow transition-transform peer-checked:translate-x-[22px]" />
            </div>
          </label>
        )}
      </div>
    </div>
  );
}
