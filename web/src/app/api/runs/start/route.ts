import { NextResponse } from "next/server";

import { runPythonJson, spawnBenchmark } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      model?: string;
      preset?: string;
      boards?: number;
    };
    if (!body.model) {
      return NextResponse.json({ error: "Model is required." }, { status: 400 });
    }
    if (body.preset === "custom" && (!body.boards || body.boards < 1)) {
      return NextResponse.json({ error: "Custom runs require a positive board count." }, { status: 400 });
    }
    const args = ["prepare-run", "--model", body.model, "--preset", body.preset ?? "smoke"];
    if (typeof body.boards === "number" && Number.isFinite(body.boards)) {
      args.push("--boards", String(body.boards));
    }
    const run = runPythonJson<{ id: string }>(args);
    spawnBenchmark(run.id);
    return NextResponse.json(run);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to start run." },
      { status: 500 },
    );
  }
}
