import type { Metadata, Viewport } from "next";
import { Nav } from "@/components/Nav";
import { pageClass, shellClass } from "@/lib/ui";
import "./globals.css";

export const metadata: Metadata = {
  title: "Scrabble LLM Benchmark",
  description: "Benchmark models on exact immediate-score Scrabble move finding.",
};

export const viewport: Viewport = {
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f8fb" },
    { media: "(prefers-color-scheme: dark)", color: "#0e1722" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className={shellClass}>
          <Nav />
          <main className={pageClass}>{children}</main>
        </div>
      </body>
    </html>
  );
}
