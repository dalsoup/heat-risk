import type { Metadata } from "next";
import "./globals.css";
import TabBar from "@/components/TabBar";
import HeaderWrapper from "@/components/HeaderWrapper"; // ✅ 새 클라 컴포넌트

export const metadata: Metadata = { title: "Heat Risk", description: "demo" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="min-h-dvh text-gray-900">
        <HeaderWrapper>
          {children}
        </HeaderWrapper>
      </body>
    </html>
  );
}
