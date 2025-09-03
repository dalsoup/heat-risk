"use client";

import type { LucideProps } from "lucide-react";
import {
  Coffee,
  HeartHandshake,
  ToyBrick,
  Droplet,
  Fan,
  Ticket,
} from "lucide-react";
import { useState } from "react";

type Item = {
  title: string;
  brand: string;
  price: number; // pt
  Icon: (props: LucideProps) => JSX.Element;
  iconBg: string; // 배경
  iconFg: string; // 아이콘 색
  cat: "전체" | "카페" | "외식" | "후원" | "기타";
};

const ALL_ITEMS: Item[] = [
  { title: "아이스 아메리카노", brand: "스타벅스", price: 6500, Icon: Coffee, iconBg: "bg-amber-50", iconFg: "text-amber-700", cat: "카페" },
  { title: "자립 청소년 후원", brand: "굿네이버스", price: 1000, Icon: HeartHandshake, iconBg: "bg-rose-50", iconFg: "text-rose-600", cat: "후원" },
  { title: "기적 인형 키링", brand: "굿즈샵", price: 5000, Icon: ToyBrick, iconBg: "bg-violet-50", iconFg: "text-violet-700", cat: "기타" },
  { title: "삼다수 500ml", brand: "CU", price: 1500, Icon: Droplet, iconBg: "bg-sky-50", iconFg: "text-sky-700", cat: "외식" },
  { title: "기적 손풍기", brand: "굿즈샵", price: 20000, Icon: Fan, iconBg: "bg-teal-50", iconFg: "text-teal-700", cat: "기타" },
  { title: "식수 후원", brand: "유니세프", price: 5000, Icon: Droplet, iconBg: "bg-blue-50", iconFg: "text-blue-700", cat: "후원" },
];

const CATS: Item["cat"][] = ["전체", "카페", "외식", "후원", "기타"];

export default function PointsPage() {
  const [cat, setCat] = useState<Item["cat"]>("전체");

  const items = ALL_ITEMS.filter(i => (cat === "전체" ? true : i.cat === cat));

  return (
    <div className="mt-3 px-4 pb-28 space-y-5">
      {/* 보유 포인트 배너 */}
      <section className="rounded-2xl bg-white shadow-sm border border-gray-100 p-4">
        <button
          className="w-full flex items-center justify-between"
          // onClick={() => router.push("/points/wallet")}
          aria-label="포인트 지갑 열기"
        >
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-gray-200 text-gray-700 font-extrabold">P</span>
            <div className="text-[17px] font-bold">12,056원</div>
          </div>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400" />
          </svg>
        </button>
      </section>

      {/* 섹션 헤더 */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-gray-200">
            <Ticket className="w-4 h-4 text-gray-600" />
          </span>
          <h2 className="text-[17px] font-extrabold">포인트로 쇼핑하기</h2>
        </div>
        <div className="text-xs text-gray-500">정렬: 오름차순</div>
      </header>

      {/* 카테고리 칩 */}
      <div className="flex items-center gap-2">
        {CATS.map((c) => {
          const active = c === cat;
          return (
            <button
              key={c}
              onClick={() => setCat(c)}
              className={`flex-1 h-8 rounded-full text-sm transition
                ${active ? "bg-gray-800 text-white" : "bg-gray-200 text-gray-700"}`}
            >
              {c}
            </button>
          );
        })}
      </div>

      {/* 아이템 그리드 (모바일 2열) */}
      <div className="grid grid-cols-2 gap-4">
        {items.map(({ title, brand, price, Icon, iconBg, iconFg }, i) => (
          <div
            key={i}
            className="rounded-2xl bg-white shadow-sm border border-gray-100 overflow-hidden"
          >
            {/* 상단 아이콘 영역 */}
            <div className="h-24 flex items-center justify-center">
              <span className={`w-16 h-16 rounded-2xl ${iconBg} flex items-center justify-center`}>
                <Icon className={iconFg} strokeWidth={1.8} size={30} />
              </span>
            </div>

            {/* 정보 */}
            <div className="px-3 pb-3">
              <div className="text-[11px] text-gray-500">{brand}</div>
              <div className="text-[13px] font-semibold leading-snug line-clamp-2">{title}</div>
              <div className="mt-1 text-right text-indigo-700 font-extrabold">
                {price.toLocaleString()}pt
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
