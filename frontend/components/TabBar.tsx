// components/TabBar.tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "홈" },
  { href: "/points", label: "포인트" },
  { href: "/self", label: "자가진단" },     
  { href: "/mypage", label: "마이페이지" },
];

export default function TabBar() {
  const pathname = usePathname();
  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(href + "/");

  return (
    <div className="h-16 bg-white rounded-2xl shadow-lg border border-gray-100 grid grid-cols-4">
      {items.map((it) => {
        const active = isActive(it.href);
        return (
          <Link
            key={it.href}
            href={it.href}
            aria-current={active ? "page" : undefined}
            className={`flex flex-col items-center justify-center text-xs transition-colors ${
              active ? "text-gray-900 font-semibold" : "text-gray-500"
            }`}
          >
            <div className={`w-6 h-6 rounded-full border flex items-center justify-center mb-1 ${active ? "border-gray-700" : "border-gray-300"}`}>N</div>
            {it.label}
          </Link>
        );
      })}
    </div>
  );
}
