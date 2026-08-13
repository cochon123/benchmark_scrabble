from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .runner import parse_tool_payload


def error_category(error: str | None) -> str:
    if error is None:
        return "legal"
    text = error.lower()
    if "rack cannot supply" in text:
        return "rack"
    if "out of bounds" in text:
        return "bounds"
    if "collides" in text:
        return "collision"
    if "no new tiles" in text:
        return "empty"
    if "gap" in text or "same row or column" in text:
        return "geometry"
    if "does not connect" in text:
        return "connection"
    if "cross word is invalid" in text:
        return "cross_word"
    if "invalid" in text:
        return "main_word"
    if "json" in text or "payload" in text or "parse" in text or "envelope" in text:
        return "format"
    return "other"


def claimed_score(raw_response: str) -> int | None:
    matches = re.findall(
        r"(?:scores?|score(?:d)?(?:\s+of)?|maximum)\s*[:=]?\s*(\d+)\s*(?:points?)?",
        raw_response,
        flags=re.IGNORECASE,
    )
    return int(matches[-1]) if matches else None


def _parseable(raw_response: str) -> bool:
    try:
        parse_tool_payload(raw_response)
    except Exception:
        return False
    return True


def summarize_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    results = list(payload.get("results", []))
    max_attempts = max((len(item.get("attempts", [])) for item in results), default=0)
    attempt_rows = []
    for attempt_number in range(1, max_attempts + 1):
        attempts = [
            item["attempts"][attempt_number - 1]
            for item in results
            if len(item.get("attempts", [])) >= attempt_number
        ]
        legal = sum(item.get("error") is None for item in attempts)
        parseable = sum(_parseable(str(item.get("raw_response", ""))) for item in attempts)
        errors = Counter(error_category(item.get("error")) for item in attempts)
        cumulative_legal = sum(
            any(attempt.get("error") is None for attempt in item.get("attempts", [])[:attempt_number])
            for item in results
        )
        attempt_rows.append(
            {
                "attempt": attempt_number,
                "boards_attempted": len(attempts),
                "legal_this_attempt": legal,
                "legal_this_attempt_pct": 100 * legal / len(attempts) if attempts else 0,
                "cumulative_legal": cumulative_legal,
                "cumulative_legal_pct": 100 * cumulative_legal / len(results) if results else 0,
                "parseable": parseable,
                "parseable_pct": 100 * parseable / len(attempts) if attempts else 0,
                "outcomes": dict(sorted(errors.items())),
            }
        )

    final_legal = [item for item in results if item.get("error") is None]
    optimal = [item for item in final_legal if bool(item.get("is_optimal"))]
    recovered = [
        item
        for item in final_legal
        if len(item.get("attempts", [])) > 1
        and item["attempts"][0].get("error") is not None
    ]
    claims = []
    for item in final_legal:
        claim = claimed_score(str(item.get("raw_response", "")))
        if claim is not None:
            claims.append((claim, int(item.get("score", 0))))
    final_errors = Counter(error_category(item.get("error")) for item in results)
    raw_points = sum(int(item.get("score", 0)) for item in results)
    optimal_points = sum(int(item.get("optimal_score", 0)) for item in results)
    return {
        "boards": len(results),
        "attempts": attempt_rows,
        "final": {
            "legal": len(final_legal),
            "legal_pct": 100 * len(final_legal) / len(results) if results else 0,
            "optimal": len(optimal),
            "optimal_pct": 100 * len(optimal) / len(results) if results else 0,
            "optimal_given_legal_pct": 100 * len(optimal) / len(final_legal) if final_legal else 0,
            "raw_points": raw_points,
            "optimal_points": optimal_points,
            "score_pct": 100 * raw_points / optimal_points if optimal_points else 0,
            "recovered_after_rejection": len(recovered),
            "outcomes": dict(sorted(final_errors.items())),
            "score_claims": len(claims),
            "exact_score_claims": sum(claim == actual for claim, actual in claims),
            "mean_absolute_score_claim_error": (
                sum(abs(claim - actual) for claim, actual in claims) / len(claims)
                if claims
                else None
            ),
        },
    }
