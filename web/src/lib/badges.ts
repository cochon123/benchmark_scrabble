export type BadgeTheme = {
  label: string;
  bg: string;
  fg: string;
  logoSrc?: string;
};

const palette: Record<string, BadgeTheme> = {
  openai: { label: "OA", bg: "#17324d", fg: "#edf6ff", logoSrc: "/logos/openai_logo.png" },
  anthropic: { label: "AN", bg: "#25405d", fg: "#edf6ff", logoSrc: "/logos/claude_logo.png" },
  google: { label: "GO", bg: "#284c73", fg: "#edf6ff", logoSrc: "/logos/gemini_logo.png" },
  meta: { label: "ME", bg: "#355b87", fg: "#edf6ff", logoSrc: "/logos/meta_logo.png" },
  qwen: { label: "QW", bg: "#1f3f69", fg: "#edf6ff", logoSrc: "/logos/qwen_logo.png" },
  mistralai: { label: "MI", bg: "#274868", fg: "#edf6ff" },
  microsoft: { label: "MS", bg: "#1b3654", fg: "#edf6ff" },
  deepseek: { label: "DS", bg: "#0f2b46", fg: "#edf6ff", logoSrc: "/logos/deepseek_logo.png" },
  xai: { label: "xAI", bg: "#1f2937", fg: "#f8fbff", logoSrc: "/logos/grok_logo.png" },
  "z-ai": { label: "ZA", bg: "#213b58", fg: "#edf6ff", logoSrc: "/logos/z_ai_logo.png" },
  googlefree: { label: "GO", bg: "#284c73", fg: "#edf6ff", logoSrc: "/logos/gemini_logo.png" },
  minimax: { label: "MM", bg: "#24476a", fg: "#edf6ff", logoSrc: "/logos/minimax_logo.png" },
  moonshotai: { label: "KM", bg: "#24476a", fg: "#edf6ff", logoSrc: "/logos/kimi_logo.png" },
  nvidia: { label: "NV", bg: "#24476a", fg: "#edf6ff", logoSrc: "/logos/nvidia_logo.png" },
  xiaomi: { label: "XM", bg: "#24476a", fg: "#edf6ff", logoSrc: "/logos/xiaomi_mimo_logo.png" },
  unknown: { label: "??", bg: "#334155", fg: "#f8fbff" },
};

export function badgeTheme(companySlug: string) {
  return palette[companySlug] ?? {
    label: companySlug.slice(0, 2).toUpperCase() || "??",
    bg: "#274868",
    fg: "#edf6ff",
  };
}
