"use client";

import Link from "next/link";

export default function SelfHome() {
  const faq = [
    ["온열질환이란 무엇인가요?", "체온이 비정상적으로 상승하여 체온 조절이 제대로 이루어지지 않는 상태입니다."],
    ["어떤 증상이 나타날 수 있나요?", "두통, 어지럼증, 구역질, 과도한 발한, 근육경련, 피로감, 체온 상승 등이 나타날 수 있습니다."],
    ["누가 특히 조심해야 하나요?", "노인, 영유아, 임산부, 만성질환자, 야외근로자, 장시간 실외활동을 하는 분들."],
    ["증상이 나타나면 어떻게 해야 하나요?", "시원한 장소로 이동 → 시원한 물을 조금씩 마시기 → 젖은 수건이나 미지근한 물로 몸 식히기."],
    ["평소에 예방법은?", "갈증 전 수분 섭취, 가벼운 옷차림, 한낮 외출 자제, 무더위 쉼터 활용."],
    ["생활 속 관리법?", "규칙적인 수면, 영양 섭취, 더운 환경에서는 활동량 줄이기, 모자·양산 활용."],
    ["언제 의료기관에 가야 하나요?", "두통·구토·근육경련이 지속되거나 체온이 40℃ 이상으로 오르고 의식 저하/경련이 동반될 때."],
    ["더위를 많이 타는 직업군 대비법?", "규칙적인 휴식시간 확보, 작업 전·중·후 수분 섭취, 냉각 장비/쿨조끼 활용."],
  ];

  return (
    <div className="mt-3 px-4 pb-28 space-y-4">
      {/* 상단 CTA */}
      <Link
        href="/self/report"
        className="block rounded-2xl bg-rose-500 text-white text-center text-xl font-extrabold py-4"
      >
        온열질환 자가신고
      </Link>

      <button
        className="w-full rounded-2xl bg-gray-200 text-gray-800 text-xl font-extrabold py-4"
        disabled
        title="곧 제공됩니다"
      >
        기후보험 청구
      </button>

      {/* FAQ 카드 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">📝</div>
          <div className="text-lg font-bold">온열질환 FAQ</div>
        </div>

        <ul className="divide-y divide-gray-200">
          {faq.map(([q, a], i) => (
            <li key={i} className="py-3">
              <div className="font-extrabold">Q{i + 1}. {q}</div>
              <div className="mt-1 text-sm text-gray-600">A{i + 1}. {a}</div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
