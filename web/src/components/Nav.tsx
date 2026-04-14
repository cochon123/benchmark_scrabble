import Link from "next/link";

import { eyebrowClass, navRowClass, secondaryButtonClass, shellHeaderClass, shellTitleClass } from "@/lib/ui";

const links = [
  { href: "/", label: "Benchmark" },
  { href: "/dataset", label: "Dataset" },
  { href: "/runs", label: "Runs" },
  { href: "/runs/new", label: "New Run" },
];

export function Nav() {
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
