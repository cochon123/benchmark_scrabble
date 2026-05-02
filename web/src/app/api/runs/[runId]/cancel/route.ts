import { NextResponse } from "next/server";

import { runPythonJson } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(_: Request, { params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  try {
    const payload = runPythonJson<{ ok: boolean; run_id: string }>(["api", "cancel-run", "--run-id", runId]);
    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to cancel run." },
      { status: 500 },
    );
  }
}
