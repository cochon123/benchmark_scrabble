from __future__ import annotations

import re


def completion_loss_weights(
    content: str,
    offsets: list[tuple[int, int]],
    *,
    json_weight: float,
    placement_weight: float,
) -> list[float]:
    """Weight final tool JSON and its coordinate/letter values more heavily."""
    weights = [1.0] * len(offsets)
    json_start = content.rfind('{"tool"')
    if json_start < 0:
        return weights

    for index, (_start, end) in enumerate(offsets):
        if end > json_start:
            weights[index] = json_weight

    placement_pattern = re.compile(
        r'"(?:row|col)":\s*-?\d+|"letter":\s*"[A-Za-z]"'
    )
    for match in placement_pattern.finditer(content, json_start):
        for index, (start, end) in enumerate(offsets):
            if end > match.start() and start < match.end():
                weights[index] = placement_weight
    return weights
