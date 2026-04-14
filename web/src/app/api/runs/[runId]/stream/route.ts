import fs from "node:fs/promises";

import { getLogPath } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params;
  const encoder = new TextEncoder();
  const logPath = getLogPath(runId);

  const stream = new ReadableStream({
    start(controller) {
      let offset = 0;
      let closed = false;
      let remainder = "";

      const send = (payload: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      };

      const heartbeat = () => {
        controller.enqueue(encoder.encode(": ping\n\n"));
      };

      const timer = setInterval(async () => {
        if (closed) {
          return;
        }
        try {
          const content = await fs.readFile(logPath, "utf-8").catch(() => "");
          if (content.length > offset) {
            const chunk = content.slice(offset);
            offset = content.length;
            const merged = remainder + chunk;
            const lines = merged.split("\n");
            remainder = lines.pop() ?? "";
            for (const line of lines) {
              if (!line.trim()) {
                continue;
              }
              try {
                const payload = JSON.parse(line) as { type?: string };
                send(payload);
                if (payload.type === "run_completed" || payload.type === "run_failed") {
                  clearInterval(timer);
                  clearInterval(heartbeatTimer);
                  closed = true;
                  controller.close();
                  break;
                }
              } catch {
                remainder = `${line}\n${remainder}`;
                break;
              }
            }
          } else {
            heartbeat();
          }
        } catch (error) {
          send({ type: "stream_error", error: error instanceof Error ? error.message : "Stream failure" });
          clearInterval(timer);
          clearInterval(heartbeatTimer);
          closed = true;
          controller.close();
        }
      }, 250);

      const heartbeatTimer = setInterval(() => {
        if (!closed) {
          heartbeat();
        }
      }, 15000);

      request.signal.addEventListener("abort", () => {
        clearInterval(timer);
        clearInterval(heartbeatTimer);
        if (!closed) {
          closed = true;
          controller.close();
        }
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Cache-Control": "no-cache, no-transform",
      "Content-Type": "text/event-stream",
      Connection: "keep-alive",
    },
  });
}
