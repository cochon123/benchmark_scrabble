import { NewRunConsole } from "@/components/NewRunConsole";
import { pageClass } from "@/lib/ui";

export default function NewRunPage() {
  return (
    <div className={pageClass}>
      <NewRunConsole />
    </div>
  );
}
