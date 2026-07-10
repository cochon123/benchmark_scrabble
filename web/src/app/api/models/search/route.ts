import { NextResponse } from "next/server";

import { runPythonJson } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type SearchResult = {
  slug: string;
  name: string;
  author?: string;
  created_at?: string;
  source?: string;
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("q") ?? "";
  if (!query) {
    return NextResponse.json([]);
  }

  // Prefer the Python helper so CLI + OpenRouter stay in one place.
  try {
    const payload = runPythonJson<SearchResult[]>(["api", "search-models", "--query", query]);
    return NextResponse.json(payload);
  } catch {
    // Fallback: OpenRouter frontend search only (no CLI).
    try {
      const response = await fetch(
        `https://openrouter.ai/api/frontend/models/find?q=${encodeURIComponent(query)}`,
        { cache: "no-store" },
      );
      const payload = await response.json();
      const models = (payload?.data?.models ?? []).map((model: Record<string, unknown>) => ({
        slug: model.slug,
        name: model.name,
        author: model.author,
        created_at: model.created_at,
        source: "openrouter",
      }));
      return NextResponse.json(models);
    } catch {
      return NextResponse.json([]);
    }
  }
}
