from __future__ import annotations

import json
from typing import Any, Iterable

from .constants import BOARD_SIZE
from .lexicon import Lexicon
from .solver import Move
from .training import position_key
from .v41_training import move_plan
from .v42_training import placements_for_start


# These are ordinary, already-trained Qwen vocabulary items, not newly-added tokens.
# The corpus builder audits that every glyph is exactly one token for the selected
# tokenizer.  Reusing the same glyph in the board and action turns localization
# into a pointer/copy operation without training 225 new embedding rows.
_SQUARE_GLYPHS = (
    "一丁七万丈三与丏丐丑专且丕世丘丙业丛东丝丞丟丢两严並丧丨个丫丰串临丸丹为丽举乂乃久么义之乌乍乎乏乐乒乓乔乖乗乘乙乜九乞也习乡书乩买乱乳乸乾亂了予争事二亍于亏云互亓五井亘亚些亞亟亡亢交亥亦产亨亩享京亭亮亲亳亵亶亸亹人亿什仁仂仃仄仅仆仇仉今介仍从仑仓仔仕他仗付仙仝仞仟仡代令以仨仪仫们仰仲仳仵件价任份仿企伈伉伊伋伍伎伏伐休众优伙会伛伝伞伟传伢伣伤伥伦伧伪伫伭伯估伲伴伶伸伺似伽伾佁佃但佈位低住佐佑体佔何佖佗佘余佚佛作佝佞佟你佣佤佥佩佬佯佰佳佴佶佸佺佻佼佽"
)
SQUARE_TOKENS = tuple(_SQUARE_GLYPHS[: BOARD_SIZE * BOARD_SIZE])
if len(SQUARE_TOKENS) != BOARD_SIZE * BOARD_SIZE or len(set(SQUARE_TOKENS)) != len(
    SQUARE_TOKENS
):
    raise RuntimeError("The V11 square-token table must contain 225 unique glyphs.")


def square_token(row: int, col: int) -> str:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"Square outside board: ({row}, {col})")
    return SQUARE_TOKENS[row * BOARD_SIZE + col]


def token_square(token: str) -> tuple[int, int]:
    try:
        index = SQUARE_TOKENS.index(token)
    except ValueError as error:
        raise ValueError("Unknown square token.") from error
    return divmod(index, BOARD_SIZE)


def tied_board_text(position: dict[str, Any]) -> str:
    cells = {(int(x["row"]), int(x["col"])): x for x in position["board"]}
    rows = []
    for row in range(BOARD_SIZE):
        encoded = []
        for col in range(BOARD_SIZE):
            cell = cells.get((row, col))
            if cell is None:
                value = "."
            else:
                letter = str(cell["letter"])
                value = letter.lower() if bool(cell.get("is_blank", False)) else letter.upper()
            encoded.append(f"{square_token(row, col)}:{value}")
        rows.append(" ".join(encoded))
    return "\n".join(rows)


def action_text(plan: dict[str, Any]) -> str:
    direction = "A" if str(plan["direction"]) == "across" else "D"
    if str(plan["direction"]) not in {"across", "down"}:
        raise ValueError("Direction must be across or down.")
    return f"{str(plan['word']).upper()}|{square_token(int(plan['row']), int(plan['col']))}|{direction}"


def parse_action(text: str) -> dict[str, Any]:
    parts = text.strip().split("|")
    if len(parts) != 3:
        raise ValueError("Response must have exactly WORD|SQUARE|A_OR_D.")
    word, token, direction = parts
    if not word or any(letter < "A" or letter > "Z" for letter in word):
        raise ValueError("The word must contain uppercase A-Z only.")
    if direction not in {"A", "D"}:
        raise ValueError("Direction must be A or D.")
    row, col = token_square(token)
    return {
        "word": word,
        "row": row,
        "col": col,
        "direction": "across" if direction == "A" else "down",
    }


def action_to_placements(
    position: dict[str, Any], text: str, lexicon: Lexicon
) -> tuple[list[dict[str, Any]], int]:
    plan = parse_action(text)
    return placements_for_start(
        position,
        lexicon,
        word=str(plan["word"]),
        direction=str(plan["direction"]),
        start_row=int(plan["row"]),
        start_col=int(plan["col"]),
    )


def move_action(position: dict[str, Any], move: Move | dict[str, Any], lexicon: Lexicon) -> str:
    placements = move.placements if isinstance(move, Move) else move["placements"]
    raw = [item.to_dict() if hasattr(item, "to_dict") else item for item in placements]
    plan = move_plan(position, raw, lexicon)
    return action_text(
        {
            "word": plan["word"],
            "row": plan["start_row"],
            "col": plan["start_col"],
            "direction": plan["direction"],
        }
    )


def policy_messages(
    position: dict[str, Any], *, rejection_history: Iterable[str] = ()
) -> list[dict[str, str]]:
    system = (
        "You are a legal-move Scrabble policy. Each board square has a unique glyph. "
        "Return one legal move as WORD|GLYPH|A_OR_D, where WORD is the complete main "
        "word, GLYPH is copied from its start square, A is across, and D is down. "
        "Use rack letters only for empty squares. Lowercase board letters are existing "
        "zero-point blanks. Return only the action, with no prose or whitespace."
    )
    user = f"RACK:{str(position['rack']).upper()}\nBOARD:\n{tied_board_text(position)}"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for rejection in rejection_history:
        messages.extend(
            [
                {"role": "assistant", "content": str(rejection).split("\n", 1)[0]},
                {
                    "role": "user",
                    "content": "REJECTED BY VERIFIER. Try a different legal action. No solution is supplied.",
                },
            ]
        )
    return messages


def training_record(
    position: dict[str, Any], target_action: str, *, stage: str, legal_move_count: int
) -> dict[str, Any]:
    # Assert that the target grammar points to a real square before serializing it.
    parse_action(target_action)
    return {
        "id": f"{position['id']}--v11-{stage}",
        "source_id": position["id"],
        "source_game_id": position.get("source_game_id", position["id"]),
        "position_key": position_key(position),
        "record_type": f"v11-{stage}",
        "legal_move_count": int(legal_move_count),
        "messages": policy_messages(position)
        + [{"role": "assistant", "content": target_action}],
    }


def gate_row(position: dict[str, Any], *, category: str) -> dict[str, Any]:
    return {
        "id": position["id"],
        "source_id": position["id"],
        "source_game_id": position.get("source_game_id", position["id"]),
        "category": category,
        "position": position,
        "messages": policy_messages(position),
    }


def retry_messages(row: dict[str, Any], rejected_actions: Iterable[str]) -> list[dict[str, str]]:
    return policy_messages(row["position"], rejection_history=rejected_actions)


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
