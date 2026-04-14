import { BoardCell, Placement } from "@/lib/types";

const WORD_MULTIPLIERS = [
  [3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3],
  [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
  [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
  [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
  [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
  [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
  [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
  [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
  [3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3],
] as const;

const LETTER_MULTIPLIERS = [
  [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
  [1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
  [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1],
  [1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
  [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
  [1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1],
  [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
  [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1],
  [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
] as const;

function premiumSquare(row: number, col: number) {
  const word = WORD_MULTIPLIERS[row][col];
  const letter = LETTER_MULTIPLIERS[row][col];
  if (word === 3) {
    return { label: "TW", className: "bg-[#9bb8d6] text-[#425563]" };
  }
  if (word === 2) {
    return { label: "DW", className: "bg-[#d8e5f2] text-[#425563]" };
  }
  if (letter === 3) {
    return { label: "TL", className: "bg-[#6f95bd] text-[#f8fbff]" };
  }
  if (letter === 2) {
    return { label: "DL", className: "bg-[#e8f0f8] text-[#425563]" };
  }
  return null;
}

function buildMaps(board: BoardCell[], highlightPlacements: Placement[] = []) {
  const boardMap = new Map<string, BoardCell>();
  const highlightMap = new Map<string, Placement>();

  for (const cell of board) {
    boardMap.set(`${cell.row}-${cell.col}`, cell);
  }
  for (const placement of highlightPlacements) {
    highlightMap.set(`${placement.row}-${placement.col}`, placement);
  }

  return { boardMap, highlightMap };
}

export function BoardView({
  board,
  rack,
  highlightPlacements,
  highlightTone = "optimal",
}: {
  board: BoardCell[];
  rack: string;
  highlightPlacements?: Placement[];
  highlightTone?: "optimal" | "model";
}) {
  const { boardMap, highlightMap } = buildMaps(board, highlightPlacements);
  const baseCellClass =
    "grid aspect-square min-h-6 place-items-center rounded-[7px] border font-extrabold max-[840px]:min-h-5 max-[840px]:rounded-[6px]";

  return (
    <div className="grid gap-[14px]">
      <div className="grid w-full [grid-template-columns:repeat(15,minmax(0,1fr))] gap-[3px] max-[840px]:gap-[2px]">
        {Array.from({ length: 15 * 15 }).map((_, index) => {
          const row = Math.floor(index / 15);
          const col = index % 15;
          const key = `${row}-${col}`;
          const base = boardMap.get(key);
          const highlight = highlightMap.get(key);
          const premium = premiumSquare(row, col);
          const classes = [baseCellClass];

          if (highlight) {
            classes.push(
              highlightTone === "optimal"
                ? "border-[#256f3d] bg-[#2f8f4e] text-[#f7fff9]"
                : "border-[rgba(65,107,150,0.92)] bg-[rgba(65,107,150,0.78)] text-white",
            );
          } else if (base) {
            classes.push("border-[rgba(25,36,45,0.08)] bg-[rgba(154,167,177,0.55)]");
          } else if (premium) {
            classes.push("border-[rgba(25,36,45,0.08)]", premium.className);
          } else {
            classes.push("border-[rgba(25,36,45,0.08)] bg-[#fff8ea]");
          }

          return (
            <div key={key} className={classes.join(" ")}>
              <span
                className={
                  base || highlight
                    ? "text-[0.84rem] leading-none"
                    : "text-[0.52rem] font-bold leading-none tracking-[0.03em] max-[640px]:text-[0.46rem]"
                }
              >
                {highlight?.letter ?? base?.letter ?? premium?.label ?? ""}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-2.5">
        {rack.split("").map((letter, index) => (
          <span
            key={`${letter}-${index}`}
            className="min-w-[34px] rounded-xl border border-[color:var(--line)] bg-[color:var(--surface-strong)] px-2.5 py-2 text-center font-bold [font-family:var(--font-geist-mono)]"
          >
            {letter}
          </span>
        ))}
      </div>
    </div>
  );
}
