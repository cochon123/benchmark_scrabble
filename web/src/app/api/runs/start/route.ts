import { NextResponse } from "next/server";

import { runManagementEnabled } from "@/lib/deployment";
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

type RunConfig = {
  preset?: string;
  boards?: number;
  reasoningEffort?: string;
  concurrency?: number;
};

function startModel(model: string, config: RunConfig) {
  const cli = parseCliModel(model);
  if (cli) {
    const args = ["prepare-cli-run", "--agent", cli.agent, "--preset", config.preset ?? "smoke"];
    if (cli.cliModel) {
      args.push("--model", cli.cliModel);
    }
    if (config.reasoningEffort) {
      args.push("--reasoning-effort", config.reasoningEffort);
    }
    if (typeof config.boards === "number" && Number.isFinite(config.boards)) {
      args.push("--boards", String(config.boards));
    }
    const run = runPythonJson<{ id: string }>(args);
    spawnCliBenchmark(run.id, cli.agent, cli.cliModel);
    return run;
  }

  const args = ["prepare-run", "--model", model, "--preset", config.preset ?? "smoke"];
  if (config.reasoningEffort) {
    args.push("--reasoning-effort", config.reasoningEffort);
  }
  if (typeof config.boards === "number" && Number.isFinite(config.boards)) {
    args.push("--boards", String(config.boards));
  }
  const concurrency =
    typeof config.concurrency === "number" && Number.isFinite(config.concurrency)
      ? Math.max(1, Math.min(32, Math.floor(config.concurrency)))
      : 1;
  const run = runPythonJson<{ id: string }>(args);
  spawnBenchmark(run.id, concurrency);
  return run;
}

export async function POST(request: Request) {
  if (!runManagementEnabled()) {
    return NextResponse.json({ error: "Not found." }, { status: 404 });
  }
  try {
    const body = (await request.json()) as {
      model?: string;
      models?: string[];
      preset?: string;
      boards?: number;
      reasoningEffort?: string;
      concurrency?: number;
    };
    const requestedModels = Array.from(
      new Set(
        (Array.isArray(body.models) ? body.models : body.model ? [body.model] : [])
          .map((model) => model.trim())
          .filter(Boolean),
      ),
    );
    if (requestedModels.length === 0) {
      return NextResponse.json({ error: "At least one model is required." }, { status: 400 });
    }
    if (requestedModels.length > 16) {
      return NextResponse.json({ error: "A batch may contain at most 16 models." }, { status: 400 });
    }
    if (body.preset === "custom" && (!body.boards || body.boards < 1)) {
      return NextResponse.json({ error: "Custom runs require a positive board count." }, { status: 400 });
    }

    const config: RunConfig = body;
    if (!Array.isArray(body.models) && body.model) {
      return NextResponse.json(startModel(body.model, config));
    }

    const runs: Array<{ id: string; model: string }> = [];
    const errors: Array<{ model: string; error: string }> = [];
    for (const model of requestedModels) {
      try {
        const run = startModel(model, config);
        runs.push({ ...run, model });
      } catch (error) {
        errors.push({
          model,
          error: error instanceof Error ? error.message : "Unable to start run.",
        });
      }
    }
    return NextResponse.json(
      { runs, errors },
      { status: runs.length === 0 ? 409 : errors.length > 0 ? 207 : 201 },
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to start run." },
      { status: 500 },
    );
  }
}
