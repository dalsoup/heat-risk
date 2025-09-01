import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      // 프론트 /api/*  →  백엔드 /<same>
      // (API_PREFIX 없음이므로 그대로 전달)
      { source: "/api/:path*", destination: "http://127.0.0.1:8000/:path*" },
    ];
  },
};

export default nextConfig;
