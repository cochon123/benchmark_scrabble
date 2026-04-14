import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("q") ?? "";
  if (!query) {
    return NextResponse.json([]);
  }

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
  }));

  return NextResponse.json(models);
}

