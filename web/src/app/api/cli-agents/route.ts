import { NextResponse } from "next/server";

import { runPythonJson } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(runPythonJson(["api", "cli-agents"]));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to list CLI agents." },
      { status: 500 },
    );
  }
}
