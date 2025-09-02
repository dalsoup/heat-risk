export default function PointsPage() {
  const items = [
    { title: "아이스 아메리카노", brand: "스타벅스", price: 6500, img: "🥤" },
    { title: "자립 청소년 후원", brand: "굿네이버스", price: 1000, img: "💖" },
    { title: "기적 인형 키링", brand: "기적 굿즈샵", price: 5000, img: "🧸" },
    { title: "아이스시스 500ml", brand: "CU", price: 1500, img: "🫗" },
    { title: "기적 손풍기", brand: "기적 굿즈샵", price: 20000, img: "🌀" },
    { title: "식수 후원", brand: "유니세프", price: 5000, img: "💧" },
  ];

  return (
    <div className="mt-3 px-4 pb-28 space-y-4">
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="text-sm text-gray-500">보유 포인트</div>
        <div className="text-2xl font-extrabold">12,056원</div>
      </section>

      <div className="grid grid-cols-3 gap-4">
        {items.map((it, i) => (
          <div key={i} className="rounded-2xl bg-white shadow-sm border border-gray-100 overflow-hidden">
            <div className="h-24 flex items-center justify-center text-4xl">{it.img}</div>
            <div className="px-3 pb-3">
              <div className="text-xs text-gray-500">{it.brand}</div>
              <div className="text-sm font-semibold line-clamp-2">{it.title}</div>
              <div className="text-right text-indigo-700 font-extrabold mt-1">
                {it.price.toLocaleString()}pt
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
