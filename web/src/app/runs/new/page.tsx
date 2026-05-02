import { ActiveRunsList } from "@/components/ActiveRunsList";
import { NewRunConsole } from "@/components/NewRunConsole";
import { pageClass } from "@/lib/ui";

export default function NewRunPage() {
  return (
    <div className={pageClass}>
      <ActiveRunsList />
      <NewRunConsole />
    </div>
  );
}
