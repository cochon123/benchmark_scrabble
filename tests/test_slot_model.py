import torch

from scrabble_bench.slot_model import LocalSlotProbe, SLOT_COUNT, SpatialSlotModel, decode_slot, factor_metrics, set_nll, slot_index


def test_slot_round_trip():
    for direction in ("across", "down"):
        index = slot_index(3, 11, direction)
        assert index < SLOT_COUNT
        decoded = decode_slot(index)
        assert decoded == {"row": 3, "col": 11, "direction": direction}


def test_set_nll_accepts_any_tied_optimum():
    logits = torch.tensor([[0.0, 4.0, 4.0, -2.0]])
    one = torch.tensor([[False, True, False, False]])
    two = torch.tensor([[False, True, True, False]])
    # Adding a second equally likely optimum must not change the normalized
    # probability assigned to the tied optimal set.
    assert torch.isclose(set_nll(logits, two), set_nll(logits, one))


def test_local_probe_slot_order_and_factor_metrics():
    probe = LocalSlotProbe(4)
    encoded = torch.zeros(1, 225, 4)
    logits = probe(encoded)
    assert logits.shape == (1, 450)
    targets = torch.zeros(1, 450, dtype=torch.bool)
    targets[0, slot_index(3, 11, "down")] = True
    oracle = torch.full((1, 450), -10.0)
    oracle[0, slot_index(3, 11, "down")] = 10.0
    assert factor_metrics(oracle, targets) == {
        "row_top1_pct": 100.0, "col_top1_pct": 100.0, "direction_top1_pct": 100.0,
    }


def test_spatial_model_emits_slot_logits():
    model = SpatialSlotModel(d_model=16, nhead=4, layers=1)
    output = model(torch.zeros(2, 225, dtype=torch.long), torch.zeros(2, 225, dtype=torch.long), torch.zeros(2, 27))
    assert output.shape == (2, SLOT_COUNT)
