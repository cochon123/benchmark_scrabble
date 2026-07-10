import { NextResponse } from "next/server";

import { runPythonJson, spawnBenchmark, spawnCliBenchmark } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function parseCliModel(model: string): { agent: string; cliModel?: string } | null {
  const parts = model.trim().split("/");
  if (parts.length < 2 || parts[0] !== "cli") {
    return null;
  }
  const agent = parts[1];
  const cliModel = parts.length > 2 ? parts.slice(2).join("/") : undefined;
  return { agent, cliModel };
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      model?: string;
      preset?: string;
      boards?: number;
      reasoningEffort?: string;
      concurrency?: number;
    };
    if (!body.model) {
      return NextResponse.json({ error: "Model is required." }, { status: 400 });
    }
    if (body.preset === "custom" && (!body.boards || body.boards < 1)) {
      return NextResponse.json({ error: "Custom runs require a positive board count." }, { status: 400 });
    }

    const cli = parseCliModel(body.model);
    if (cli) {
      const args = ["prepare-cli-run", "--agent", cli.agent, "--preset", body.preset ?? "smoke"];
      if (cli.cliModel) {
        args.push("--model", cli.cliModel);
      }
      if (body.reasoningEffort) {
        args.push("--reasoning-effort", body.reasoningEffort);
      }
      if (typeof body.boards === "number" && Number.isFinite(body.boards)) {
        args.push("--boards", String(body.boards));
      }
      const run = runPythonJson<{ id: string }>(args);
      spawnCliBenchmark(run.id, cli.agent, cli.cliModel);
      return NextResponse.json(run);
    }

    const args = ["prepare-run", "--model", body.model, "--preset", body.preset ?? "smoke"];
    if (body.reasoningEffort) {
      args.push("--reasoning-effort", body.reasoningEffort);
    }
    if (typeof body.boards === "number" && Number.isFinite(body.boards)) {
      args.push("--boards", String(body.boards));
    }
    const concurrency =
      typeof body.concurrency === "number" && Number.isFinite(body.concurrency)
        ? Math.max(1, Math.min(32, Math.floor(body.concurrency)))
        : 1;
    const run = runPythonJson<{ id: string }>(args);
    spawnBenchmark(run.id, concurrency);
    return NextResponse.json(run);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to start run." },
      { status: 500 },
    );
  }
}
