import type { Metadata } from "next";
import "./globals.css";
import TabBar from "@/components/TabBar";

export const metadata: Metadata = { title: "Heat Risk", description: "demo" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="min-h-dvh bg-[#f2f3f5] text-gray-900">
        <div className="mx-auto w-full max-w-[420px] min-h-dvh bg-[#f2f3f5] flex flex-col">
          {/* 상단 상태바 느낌 여백 */}
          <div className="h-5" />

          {/* 상단 위치 */}
          <div className="px-5">
            <div className="text-sm text-gray-500">현재 위치</div>
            <div className="mt-1 text-lg font-bold leading-tight">청운효자동, 서울</div>
          </div>

          {/* 본문 (탭바와 겹치지 않게 하단 패딩) */}
          <main className="flex-1 px-4 pb-28">{children}</main>

          {/* 하단 탭바 */}
          <nav className="fixed inset-x-0 bottom-0 z-50">
            <div className="mx-auto w-full max-w-[420px] px-4 pb-[env(safe-area-inset-bottom)]">
              <TabBar />
            </div>
          </nav>
        </div>
      </body>
    </html>
  );
}
