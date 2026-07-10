import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(process.cwd(), "..");
const pythonArgs = ["-m", "scrabble_bench"];

function runPython(args: string[]) {
  const result = spawnSync("python3", [...pythonArgs, ...args], {
    cwd: repoRoot,
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024,
  });

  if (result.status !== 0) {
    const details = result.stderr?.trim() || result.stdout?.trim() || "Unknown Python command failure.";
    throw new Error(details);
  }

  return result.stdout.trim();
}

export function runPythonJson<T>(args: string[]): T {
  const output = runPython(args);
  return JSON.parse(output) as T;
}

export function spawnBenchmark(runId: string, concurrency = 1) {
  const child = spawn("python3", [...pythonArgs, "run", "--run-id", runId, "--concurrency", String(concurrency)], {
    cwd: repoRoot,
    detached: true,
    stdio: "ignore",
  });
  child.unref();
}

export function spawnCliBenchmark(runId: string, agent: string, model?: string | null) {
  const args = [...pythonArgs, "cli-run", "--run-id", runId, "--agent", agent];
  if (model) {
    args.push("--model", model);
  }
  const child = spawn("python3", args, {
    cwd: repoRoot,
    detached: true,
    stdio: "ignore",
  });
  child.unref();
}

export function exportCsv(kind: "all" | "run", runId?: string) {
  const args = ["export", "--kind", kind];
  if (kind === "run" && runId) {
    args.push("--run-id", runId);
  }
  const outputPath = runPython(args);
  return outputPath;
}

export function getLogPath(runId: string) {
  return path.join(repoRoot, "data", "logs", `run_${runId}.jsonl`);
}

export function readFileIfExists(filePath: string) {
  try {
    return fs.readFileSync(filePath, "utf-8");
  } catch {
    return "";
  }
}

export { repoRoot };
