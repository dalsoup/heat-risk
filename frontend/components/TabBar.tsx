// components/TabBar.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { LucideProps } from "lucide-react";
import { Home, ShoppingCart, ClipboardCheck, UserRound } from "lucide-react";

type Item = {
  href: string;
  label: string;
  Icon: (props: LucideProps) => JSX.Element;
};

const items: Item[] = [
  { href: "/",       label: "홈",       Icon: Home },
  { href: "/points", label: "포인트",   Icon: ShoppingCart },
  { href: "/self",   label: "자가진단", Icon: ClipboardCheck }, // /self/* 하위 경로 포함
  { href: "/mypage", label: "마이페이지", Icon: UserRound },
];

export default function TabBar() {
  const pathname = usePathname();
  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(href + "/");

  return (
    <nav
      aria-label="하단 탭바"
      className="h-16 bg-white rounded-2xl shadow-lg border border-gray-100 grid grid-cols-4"
      role="navigation"
    >
      {items.map(({ href, label, Icon }) => {
        const active = isActive(href);
        const color = active ? "text-emerald-700" : "text-gray-500";
        const ring  = active ? "border-emerald-700" : "border-gray-300";

        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={`flex flex-col items-center justify-center text-[12px] font-semibold transition-colors ${color}`}
          >
            <span className={`mb-1 inline-flex items-center justify-center w-7 h-7 rounded-full border ${ring}`}>
              <Icon size={22} strokeWidth={1.8} className="shrink-0" />
            </span>
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
