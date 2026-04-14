import fs from "node:fs/promises";
import path from "node:path";

import { exportCsv } from "@/lib/server";

export const runtime = "nodejs";

export async function GET() {
  const csvPath = exportCsv("all");
  const data = await fs.readFile(csvPath);
  return new Response(data, {
    headers: {
      "Content-Type": "text/csv",
      "Content-Disposition": `attachment; filename="${path.basename(csvPath)}"`,
    },
  });
}
