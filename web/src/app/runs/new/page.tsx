import { notFound } from "next/navigation";

import { ActiveRunsList } from "@/components/ActiveRunsList";
import { NewRunConsole } from "@/components/NewRunConsole";
import { runManagementEnabled } from "@/lib/deployment";
import { pageClass } from "@/lib/ui";

export default function NewRunPage() {
  if (!runManagementEnabled()) {
    notFound();
  }
  return (
    <div className={pageClass}>
      <ActiveRunsList />
      <NewRunConsole />
    </div>
  );
}
