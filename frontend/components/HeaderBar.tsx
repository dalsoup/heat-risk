"use client";

import LocationBar from "@/components/LocationBar";

/**
 * 상단 위치 바.
 * - forceBg로 배경을 강제 지정 (래퍼가 경로/레벨에 따라 넘겨줌)
 */
export default function HeaderBar({ forceBg }: { forceBg: string }) {
  return <LocationBar defaultText="이촌동, 서울" bg={forceBg} showToggle={true} />;
}
