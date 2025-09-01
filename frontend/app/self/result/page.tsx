"use client";

type Result = {
  level: "mild" | "severe";
  headline: string;
  recommendations: string[];
  actions: { shelter: boolean; hospital: boolean; call119: boolean };
};

export default function ResultPage({ searchParams }: { searchParams: { data?: string } }) {
  let res: Result | null = null;
  try {
    if (searchParams?.data) res = JSON.parse(decodeURIComponent(searchParams.data));
  } catch {/* noop */}

  if (!res) {
    return (
      <div className="p-6">
        <div className="text-sm text-gray-600">결과 정보가 없습니다. 자가신고부터 진행해주세요.</div>
      </div>
    );
  }

  const isMild = res.level === "mild";
  const bg = isMild ? "bg-teal-50" : "bg-rose-50";
  const accent = isMild ? "text-blue-800" : "text-red-700";

  return (
    <div className={`${bg} min-h-[calc(100dvh-64px)] px-4 pb-28`}>
      <div className="h-4" />
      <div className="text-3xl font-extrabold">온열질환 자가신고 결과</div>

      <section className="mt-4 rounded-2xl bg-white shadow-sm p-6">
        <div className="text-center text-2xl text-gray-700">현재 상태는</div>
        <div className={`mt-4 text-center text-5xl font-extrabold ${accent}`}>
          {res.headline}
        </div>

        <div className="mt-6 text-center text-gray-600 leading-relaxed">
          {isMild ? "즉각적 심각 위험은 아니지만, 아래 권장 사항을 따라주세요." : "즉시 대응이 필요합니다."}
        </div>

        <hr className="my-6" />

        <div className="text-xl font-bold mb-2">권장 행동</div>
        <ul className="list-disc pl-5 space-y-2">
          {res.recommendations.map((t, i) => (
            <li key={i} className="text-gray-800">{t}</li>
          ))}
        </ul>

        <div className="mt-6 grid grid-cols-2 gap-3">
          {res.actions.shelter && (
            <button className="h-14 rounded-xl bg-emerald-800 text-white font-extrabold">인근 쉼터 찾기</button>
          )}
          {res.actions.hospital && (
            <button className="h-14 rounded-xl bg-indigo-900 text-white font-extrabold">병의원 찾기</button>
          )}
          {res.actions.call119 && (
            <button className="col-span-2 h-14 rounded-xl bg-red-600 text-white text-xl font-extrabold">119 신고</button>
          )}
        </div>
      </section>
    </div>
  );
}
