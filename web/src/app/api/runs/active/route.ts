import { NextResponse } from "next/server";

import { runManagementEnabled } from "@/lib/deployment";
import { getActiveRuns } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  if (!runManagementEnabled()) {
    return NextResponse.json({ error: "Not found." }, { status: 404 });
  }
  return NextResponse.json(getActiveRuns());
}
