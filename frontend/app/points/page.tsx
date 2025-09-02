"use client";

import type { LucideProps } from "lucide-react";
import {
  Coffee,
  HeartHandshake,
  ToyBrick,
  Droplet,
  Fan,
} from "lucide-react";

type Item = {
  title: string;
  brand: string;
  price: number;
  Icon: (props: LucideProps) => JSX.Element;
  iconBg: string;   // 배경
  iconFg: string;   // 아이콘 색
};

export default function PointsPage() {
  const items: Item[] = [
    { title: "아메리카노",    brand: "스타벅스",   price: 6500,  Icon: Coffee,         iconBg: "bg-amber-50",  iconFg: "text-amber-700" },
    { title: "청소년 후원",   brand: "굿네이버스", price: 1000,  Icon: HeartHandshake, iconBg: "bg-rose-50",    iconFg: "text-rose-600" },
    { title: "인형 키링",     brand: "굿즈샵",     price: 5000,  Icon: ToyBrick,       iconBg: "bg-violet-50",  iconFg: "text-violet-700" },
    { title: "삼다수 500ml",  brand: "CU",        price: 1500,  Icon: Droplet,        iconBg: "bg-sky-50",     iconFg: "text-sky-700" },
    { title: "손풍기",        brand: "굿즈샵",     price: 20000, Icon: Fan,            iconBg: "bg-teal-50",    iconFg: "text-teal-700" },
    { title: "식수 후원",     brand: "유니세프",   price: 5000,  Icon: Droplet,        iconBg: "bg-blue-50",    iconFg: "text-blue-700" },
  ];

  return (
    <div className="mt-3 px-4 pb-28 space-y-4">
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <div className="text-sm text-gray-500">보유 포인트</div>
        <div className="text-2xl font-extrabold">12,056원</div>
      </section>

      <div className="grid grid-cols-3 gap-4">
        {items.map(({ title, brand, price, Icon, iconBg, iconFg }, i) => (
          <div
            key={i}
            className="rounded-2xl bg-white shadow-sm border border-gray-100 overflow-hidden"
          >
            {/* 상단 아이콘 영역 */}
            <div className="h-24 flex items-center justify-center">
              <span className={`w-14 h-14 rounded-2xl ${iconBg} flex items-center justify-center`}>
                <Icon size={28} strokeWidth={1.8} className={iconFg} />
              </span>
            </div>

            {/* 정보 */}
            <div className="px-3 pb-3">
              <div className="text-xs text-gray-500">{brand}</div>
              <div className="text-sm font-semibold line-clamp-2">{title}</div>
              <div className="text-right text-indigo-700 font-extrabold mt-1">
                {price.toLocaleString()}pt
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
