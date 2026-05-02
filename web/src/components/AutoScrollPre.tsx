"use client";

import { useEffect, useRef } from "react";

export function AutoScrollPre({ children, className }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [children]);

  return (
    <pre ref={ref} className={className}>
      {children}
    </pre>
  );
}
