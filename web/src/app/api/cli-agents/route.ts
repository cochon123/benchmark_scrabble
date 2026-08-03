import { NextResponse } from "next/server";

import { runManagementEnabled } from "@/lib/deployment";
import { runPythonJson } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  if (!runManagementEnabled()) {
    return NextResponse.json({ error: "Not found." }, { status: 404 });
  }
  try {
    return NextResponse.json(runPythonJson(["api", "cli-agents"]));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to list CLI agents." },
      { status: 500 },
    );
  }
}
