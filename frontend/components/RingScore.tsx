"use client";

type Props = {
  value: number;        // 0~100
  size?: number;        // px, default 140
  stroke?: number;      // 두께, default 14
  track?: string;       // 트랙 색
  color?: string;       // 진행 색
};

export default function RingScore({
  value,
  size = 140,
  stroke = 14,
  track = "#e5e7eb", // gray-200
  color = "#b91c1c", // red-700
}: Props) {
  const v = Math.max(0, Math.min(100, value));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = (v / 100) * c;

  // 시작 각도를 살짝 시안처럼 우측 상단에서 시작되게  -30deg 회전
  const rotate = -30;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="block">
        {/* 트랙 */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={track}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          opacity={0.9}
        />
        {/* 진행 아크 */}
        <g style={{ transform: `rotate(${rotate}deg)`, transformOrigin: "50% 50%" }}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            stroke={color}
            strokeWidth={stroke}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${c - dash}`}
          />
        </g>
      </svg>

      {/* 중앙 숫자 */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-[44px] leading-none font-extrabold tracking-tight">
          {v}
          <span className="ml-1 text-2xl font-bold">점</span>
        </div>
      </div>
    </div>
  );
}
