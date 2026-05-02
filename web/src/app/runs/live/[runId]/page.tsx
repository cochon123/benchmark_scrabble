import { notFound } from "next/navigation";

import { getRun } from "@/lib/store";
import { RunLive } from "@/components/RunLive";

export const dynamic = "force-dynamic";

export default async function LiveRunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const run = getRun(runId);
  if (!run) {
    notFound();
  }
  return <RunLive run={run} />;
}
