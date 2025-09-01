"use client";

type Props = {
  min: number;
  max: number;
  range: [number, number]; // 강조 구간
  color?: string;          // 강조 색(hex)
};

export default function SegmentBar({ min, max, range, color = "#f43f5e" }: Props) {
  const width = 220;
  const h = 10;
  const total = max - min;
  const startPct = ((range[0] - min) / total) * 100;
  const endPct = ((range[1] - min) / total) * 100;

  return (
    <div className="w-full">
      <div className="h-2 rounded-full bg-gray-300/70 relative overflow-hidden" style={{ height: h }}>
        <div
          className="absolute h-full rounded-full"
          style={{ left: `${startPct}%`, width: `${endPct - startPct}%`, background: color }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-gray-500">
        <span>{min}bpm</span>
        <span>{max}bpm</span>
      </div>
    </div>
  );
}
