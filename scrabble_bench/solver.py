from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .constants import ALPHABET, BINGO_BONUS, BOARD_SIZE, CENTER, LETTER_MULTIPLIERS, LETTER_VALUES, WORD_MULTIPLIERS
from .lexicon import Lexicon, TrieNode
from .models import Move, Placement


@dataclass(frozen=True)
class BoardTile:
    letter: str
    is_blank: bool = False


Grid = list[list[BoardTile | None]]


def empty_grid() -> Grid:
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def grid_from_position(board_cells: list[dict]) -> Grid:
    grid = empty_grid()
    for cell in board_cells:
        tile = BoardTile(cell["letter"].upper(), bool(cell.get("is_blank", False)))
        grid[cell["row"]][cell["col"]] = tile
    return grid


def grid_to_cells(grid: Grid) -> list[dict]:
    cells: list[dict] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            tile = grid[row][col]
            if tile is None:
                continue
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "letter": tile.letter,
                    "is_blank": tile.is_blank,
                    "is_existing": True,
                }
            )
    return cells


def is_board_empty(grid: Grid) -> bool:
    return all(tile is None for row in grid for tile in row)


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def neighbors(row: int, col: int) -> list[tuple[int, int]]:
    points = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc):
            points.append((nr, nc))
    return points


def build_rack_counter(rack: str) -> Counter[str]:
    return Counter(letter.upper() for letter in rack)


def tile_value(letter: str, is_blank: bool) -> int:
    return 0 if is_blank else LETTER_VALUES[letter]


def read_word(grid: Grid, row: int, col: int, dr: int, dc: int) -> list[tuple[int, int, BoardTile]]:
    while in_bounds(row - dr, col - dc) and grid[row - dr][col - dc] is not None:
        row -= dr
        col -= dc
    word: list[tuple[int, int, BoardTile]] = []
    while in_bounds(row, col) and grid[row][col] is not None:
        tile = grid[row][col]
        assert tile is not None
        word.append((row, col, tile))
        row += dr
        col += dc
    return word


def score_word(
    word_tiles: list[tuple[int, int, BoardTile]],
    new_positions: set[tuple[int, int]],
) -> int:
    subtotal = 0
    word_multiplier = 1
    for row, col, tile in word_tiles:
        value = tile_value(tile.letter, tile.is_blank)
        if (row, col) in new_positions:
            subtotal += value * LETTER_MULTIPLIERS[row][col]
            word_multiplier *= WORD_MULTIPLIERS[row][col]
        else:
            subtotal += value
    return subtotal * word_multiplier


def word_text(word_tiles: list[tuple[int, int, BoardTile]]) -> str:
    return "".join(tile.letter for _, _, tile in word_tiles)


def normalize_placements(raw_placements: Iterable[dict], grid: Grid) -> list[Placement]:
    deduped: dict[tuple[int, int], Placement] = {}
    for raw in raw_placements:
        row = int(raw["row"])
        col = int(raw["col"])
        letter = str(raw["letter"]).strip().upper()
        if not in_bounds(row, col):
            raise ValueError(f"Placement out of bounds: ({row}, {col})")
        if len(letter) != 1 or letter < "A" or letter > "Z":
            raise ValueError(f"Invalid placement letter: {letter!r}")
        existing = grid[row][col]
        if existing is not None:
            if existing.letter != letter:
                raise ValueError(f"Placement collides with existing tile at ({row}, {col})")
            continue
        deduped[(row, col)] = Placement(row=row, col=col, letter=letter)
    placements = sorted(deduped.values(), key=lambda item: (item.row, item.col, item.letter))
    if not placements:
        raise ValueError("No new tiles were provided.")
    if len(placements) > 7:
        raise ValueError("A Scrabble move may not place more than 7 tiles.")
    return placements


def assign_blanks(placements: list[Placement], rack: str) -> list[Placement]:
    rack_counter = build_rack_counter(rack)
    resolved: list[Placement] = []
    for placement in placements:
        letter = placement.letter
        if rack_counter[letter] > 0:
            rack_counter[letter] -= 1
            resolved.append(placement)
            continue
        if rack_counter["?"] > 0:
            rack_counter["?"] -= 1
            resolved.append(Placement(placement.row, placement.col, placement.letter, True))
            continue
        raise ValueError(f"Rack cannot supply letter {letter!r}.")
    return resolved


def infer_orientation(placements: list[Placement]) -> str | None:
    if len(placements) == 1:
        return None
    rows = {placement.row for placement in placements}
    cols = {placement.col for placement in placements}
    if len(rows) == 1:
        return "across"
    if len(cols) == 1:
        return "down"
    raise ValueError("Placements must all be on the same row or column.")


def validate_and_score_move(
    lexicon: Lexicon,
    grid: Grid,
    rack: str,
    raw_placements: Iterable[dict],
) -> Move:
    placements = assign_blanks(normalize_placements(raw_placements, grid), rack)
    orientation = infer_orientation(placements)
    new_grid = [[grid[row][col] for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)]
    for placement in placements:
        new_grid[placement.row][placement.col] = BoardTile(placement.letter, placement.is_blank)

    board_was_empty = is_board_empty(grid)
    new_positions = {(placement.row, placement.col) for placement in placements}
    words: list[str] = []
    total_score = 0
    touched_existing = False

    if board_was_empty and CENTER not in new_positions:
        raise ValueError("The opening move must cover the center square.")

    if orientation == "across":
        row = placements[0].row
        cols = [placement.col for placement in placements]
        for col in range(min(cols), max(cols) + 1):
            if new_grid[row][col] is None:
                raise ValueError("Move has a gap in its main word.")
        main_word = read_word(new_grid, row, min(cols), 0, 1)
        main_text = word_text(main_word)
        if main_text not in lexicon.words:
            raise ValueError(f"Main word is invalid: {main_text}")
        if any((r, c) not in new_positions for r, c, _ in main_word):
            touched_existing = True
        words.append(main_text)
        total_score += score_word(main_word, new_positions)
        for placement in placements:
            cross_word = read_word(new_grid, placement.row, placement.col, 1, 0)
            if len(cross_word) <= 1:
                continue
            cross_text = word_text(cross_word)
            if cross_text not in lexicon.words:
                raise ValueError(f"Cross word is invalid: {cross_text}")
            words.append(cross_text)
            total_score += score_word(cross_word, new_positions)
            touched_existing = True
    elif orientation == "down":
        col = placements[0].col
        rows = [placement.row for placement in placements]
        for row in range(min(rows), max(rows) + 1):
            if new_grid[row][col] is None:
                raise ValueError("Move has a gap in its main word.")
        main_word = read_word(new_grid, min(rows), col, 1, 0)
        main_text = word_text(main_word)
        if main_text not in lexicon.words:
            raise ValueError(f"Main word is invalid: {main_text}")
        if any((r, c) not in new_positions for r, c, _ in main_word):
            touched_existing = True
        words.append(main_text)
        total_score += score_word(main_word, new_positions)
        for placement in placements:
            cross_word = read_word(new_grid, placement.row, placement.col, 0, 1)
            if len(cross_word) <= 1:
                continue
            cross_text = word_text(cross_word)
            if cross_text not in lexicon.words:
                raise ValueError(f"Cross word is invalid: {cross_text}")
            words.append(cross_text)
            total_score += score_word(cross_word, new_positions)
            touched_existing = True
    else:
        placement = placements[0]
        horizontal = read_word(new_grid, placement.row, placement.col, 0, 1)
        vertical = read_word(new_grid, placement.row, placement.col, 1, 0)
        horizontal_text = word_text(horizontal)
        vertical_text = word_text(vertical)
        valid_horizontal = len(horizontal) > 1 or board_was_empty
        valid_vertical = len(vertical) > 1 or board_was_empty
        if not valid_horizontal and not valid_vertical:
            raise ValueError("Single-tile move does not connect to any word.")
        seen_words: set[tuple[tuple[int, int], ...]] = set()
        for word_tiles in (horizontal, vertical):
            if len(word_tiles) <= 1 and not board_was_empty:
                continue
            text = word_text(word_tiles)
            if text not in lexicon.words:
                raise ValueError(f"Word is invalid: {text}")
            key = tuple((row, col) for row, col, _ in word_tiles)
            if key in seen_words:
                continue
            seen_words.add(key)
            words.append(text)
            total_score += score_word(word_tiles, new_positions)
            if any((row, col) not in new_positions for row, col, _ in word_tiles):
                touched_existing = True

    if not board_was_empty and not touched_existing:
        raise ValueError("Move does not connect to the existing board.")
    if len(placements) == 7:
        total_score += BINGO_BONUS
    return Move(placements=placements, score=total_score, words=sorted(set(words)))


def _cross_check_letters(
    grid: Grid,
    direction: str,
    row: int,
    col: int,
    lexicon_words: set[str],
) -> frozenset[str]:
    if direction == "across":
        dr, dc = 1, 0
    else:
        dr, dc = 0, 1
    before = []
    r, c = row - dr, col - dc
    while in_bounds(r, c) and grid[r][c] is not None:
        before.append(grid[r][c].letter)
        r -= dr
        c -= dc
    before.reverse()
    after = []
    r, c = row + dr, col + dc
    while in_bounds(r, c) and grid[r][c] is not None:
        after.append(grid[r][c].letter)
        r += dr
        c += dc
    if not before and not after:
        return frozenset(ALPHABET)
    allowed = set()
    for letter in ALPHABET:
        word = "".join(before) + letter + "".join(after)
        if word in lexicon_words:
            allowed.add(letter)
    return frozenset(allowed)


def _anchors_for_direction(grid: Grid) -> list[tuple[int, int]]:
    if is_board_empty(grid):
        return [CENTER]
    anchors: list[tuple[int, int]] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if grid[row][col] is not None:
                continue
            if any(grid[nr][nc] is not None for nr, nc in neighbors(row, col)):
                anchors.append((row, col))
    return anchors


def enumerate_moves(lexicon: Lexicon, grid: Grid, rack: str) -> list[Move]:
    rack_counter = build_rack_counter(rack)
    found: dict[tuple[tuple[int, int, str, bool], ...], Move] = {}
    for direction in ("across", "down"):
        _enumerate_direction(lexicon, grid, rack, rack_counter, direction, found)
    return sorted(
        found.values(),
        key=lambda move: (
            -move.score,
            len(move.placements),
            [(p.row, p.col, p.letter, p.is_blank) for p in move.placements],
        ),
    )


def best_moves(lexicon: Lexicon, grid: Grid, rack: str) -> list[Move]:
    moves = enumerate_moves(lexicon, grid, rack)
    if not moves:
        return []
    top_score = moves[0].score
    return [move for move in moves if move.score == top_score]


def _enumerate_direction(
    lexicon: Lexicon,
    grid: Grid,
    rack: str,
    rack_counter: Counter[str],
    direction: str,
    found: dict[tuple[tuple[int, int, str, bool], ...], Move],
) -> None:
    dr, dc = (0, 1) if direction == "across" else (1, 0)
    anchors = _anchors_for_direction(grid)

    for anchor_row, anchor_col in anchors:
        fixed = anchor_col if direction == "across" else anchor_row
        for start_offset in range(0, fixed + 1):
            start_row = anchor_row - dr * start_offset
            start_col = anchor_col - dc * start_offset
            if not in_bounds(start_row, start_col):
                break
            prev_row = start_row - dr
            prev_col = start_col - dc
            if in_bounds(prev_row, prev_col) and grid[prev_row][prev_col] is not None:
                continue
            empties_before_anchor = 0
            probe_row, probe_col = start_row, start_col
            while (probe_row, probe_col) != (anchor_row, anchor_col):
                if grid[probe_row][probe_col] is None:
                    empties_before_anchor += 1
                probe_row += dr
                probe_col += dc
            if empties_before_anchor > sum(rack_counter.values()):
                continue
            _search_from_start(
                lexicon=lexicon,
                grid=grid,
                rack_counter=rack_counter.copy(),
                direction=direction,
                row=start_row,
                col=start_col,
                anchor=(anchor_row, anchor_col),
                node=lexicon.root,
                placements=[],
                found=found,
                original_rack=rack,
            )


def _search_from_start(
    lexicon: Lexicon,
    grid: Grid,
    rack_counter: Counter[str],
    direction: str,
    row: int,
    col: int,
    anchor: tuple[int, int],
    node: TrieNode,
    placements: list[Placement],
    found: dict[tuple[tuple[int, int, str, bool], ...], Move],
    original_rack: str,
) -> None:
    dr, dc = (0, 1) if direction == "across" else (1, 0)
    anchor_used = any((placement.row, placement.col) == anchor for placement in placements)

    if node.is_word and anchor_used:
        next_row, next_col = row, col
        if not in_bounds(next_row, next_col) or grid[next_row][next_col] is None:
            try:
                move = validate_and_score_move(
                    lexicon,
                    grid,
                    original_rack,
                    [placement.to_dict() for placement in placements],
                )
            except ValueError:
                move = None
            if move is not None:
                found[move.key()] = move

    if not in_bounds(row, col):
        return
    board_tile = grid[row][col]
    if board_tile is not None:
        child = node.children.get(board_tile.letter)
        if child is None:
            return
        _search_from_start(
            lexicon,
            grid,
            rack_counter,
            direction,
            row + dr,
            col + dc,
            anchor,
            child,
            placements,
            found,
            original_rack,
        )
        return

    allowed = _cross_check_letters(grid, direction, row, col, lexicon.words)
    for letter in ALPHABET:
        child = node.children.get(letter)
        if child is None or letter not in allowed:
            continue
        if rack_counter[letter] > 0:
            next_counter = rack_counter.copy()
            next_counter[letter] -= 1
            next_placements = placements + [Placement(row, col, letter)]
        elif rack_counter["?"] > 0:
            next_counter = rack_counter.copy()
            next_counter["?"] -= 1
            next_placements = placements + [Placement(row, col, letter, True)]
        else:
            continue
        _search_from_start(
            lexicon,
            grid,
            next_counter,
            direction,
            row + dr,
            col + dc,
            anchor,
            child,
            next_placements,
            found,
            original_rack,
        )
