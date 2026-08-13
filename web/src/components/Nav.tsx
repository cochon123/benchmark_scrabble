import Link from "next/link";

import { runManagementEnabled } from "@/lib/deployment";
import { eyebrowClass, navRowClass, secondaryButtonClass, shellHeaderClass, shellTitleClass } from "@/lib/ui";

const publicLinks = [
  { href: "/", label: "Benchmark" },
  { href: "/report", label: "Experiment report" },
  { href: "/dataset", label: "Dataset" },
];

const managementLinks = [
  { href: "/runs", label: "Runs" },
  { href: "/runs/new", label: "New Run" },
];

export function Nav() {
  const links = runManagementEnabled() ? [...publicLinks, ...managementLinks] : publicLinks;
  return (
    <header className={shellHeaderClass}>
      <div>
        <p className={eyebrowClass}>Scrabble LLM Benchmark</p>
        <h1 className={shellTitleClass}>Immediate-score benchmark for model move quality.</h1>
      </div>
      <nav className={navRowClass}>
        {links.map((link) => (
          <Link key={link.href} href={link.href} className={secondaryButtonClass}>
            {link.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
