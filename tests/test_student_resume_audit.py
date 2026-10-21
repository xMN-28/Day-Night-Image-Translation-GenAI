import pytest
import torch

from daynight.audit_student_resume import compare_state


def test_audit_compares_nested_optimizer_and_rng_state():
    state = {"a": [torch.ones(2), (torch.tensor(3), None)], "counter": 5}
    assert max(compare_state(state, state)) == 0


def test_audit_does_not_hide_failed_tolerance():
    errors, differences = [], []
    compare_state(torch.zeros(2), torch.ones(2), "weights", differences, errors)
    assert errors and differences == [1]
    with pytest.raises(AssertionError):
        compare_state(torch.zeros(2), torch.ones(2))
