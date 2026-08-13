import importlib.util
from pathlib import Path


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_v8_candidate_ranking.py"
_SPEC = importlib.util.spec_from_file_location("audit_v8_candidate_ranking", _PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
select_game_separated = _MODULE.select_game_separated


def test_select_game_separated_balances_bands_and_games():
    rows = [
        {"id": f"{game}-{band}", "source_game_id": game, "band_ply": band}
        for game in ("a", "b", "c", "d") for band in (6, 7)
    ]
    selected = select_game_separated(rows, 4)
    assert len(selected) == 4
    assert len({row["source_game_id"] for row in selected}) == 4
    assert {row["band_ply"] for row in selected} == {6, 7}
