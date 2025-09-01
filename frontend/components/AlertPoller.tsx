// frontend/components/AlertPoller.tsx
"use client";
import { useEffect, useState } from "react";

export function AlertPoller() {
  const [alert, setAlert] = useState<{has_alert:boolean; title:string; cta:string[]} | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch("/api/health/alert", { cache: "no-store" });
        if (r.ok) {
          const j = await r.json();
          if (alive && j?.has_alert) setAlert(j);
        }
      } catch {}
    };
    tick();
    const id = setInterval(tick, 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!alert) return null;
  return (
    <div className="fixed inset-0 bg-black/30 flex items-end justify-center pb-24">
      <div className="card w-full max-w-md">
        <div className="text-lg font-semibold">{alert.title}</div>
        <div className="mt-3 flex gap-2">
          {alert.cta?.includes("self_check") && (
            <a className="btn btn-primary flex-1" href="/self">자가진단</a>
          )}
          {alert.cta?.includes("find_shelter") && (
            <a className="btn btn-outline flex-1" href="#shelters">쉼터 보기</a>
          )}
        </div>
        <button className="mt-3 text-sm text-gray-500" onClick={() => setAlert(null)}>닫기</button>
      </div>
    </div>
  );
}
