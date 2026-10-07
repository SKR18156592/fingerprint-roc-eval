import numpy as np
import pytest

from fpeval.metrics import (
    equal_error_rate,
    far_frr_at,
    far_resolution,
    roc_curve,
    tar_at_far,
)


def test_far_frr_at_simple_threshold():
    genuine = [0.9, 0.8, 0.4, 0.7]
    impostor = [0.1, 0.2, 0.6, 0.3]
    far, frr = far_frr_at(genuine, impostor, 0.5)
    assert far == pytest.approx(0.25)  # 0.6 is accepted
    assert frr == pytest.approx(0.25)  # 0.4 is rejected


def test_score_equal_to_threshold_is_accepted():
    far, frr = far_frr_at([0.5], [0.5], 0.5)
    assert far == 1.0 and frr == 0.0


def test_roc_is_monotonic_and_spans_extremes():
    rng = np.random.default_rng(0)
    g, i = rng.uniform(0.3, 1, 200), rng.uniform(0, 0.7, 300)
    roc = roc_curve(g, i)
    assert np.all(np.diff(roc.far) <= 0)
    assert np.all(np.diff(roc.frr) >= 0)
    assert roc.far[0] == 1.0 and roc.frr[0] == 0.0
    assert roc.far[-1] == 0.0 and roc.frr[-1] == 1.0


def test_perfect_separation_has_zero_eer():
    res = equal_error_rate([0.8, 0.9, 0.95], [0.1, 0.2, 0.3])
    assert res.eer == pytest.approx(0.0)
    assert 0.3 <= res.threshold <= 0.8


def test_perfect_separation_threshold_is_mid_gap():
    # Max-margin choice: halfway between best impostor and worst genuine.
    res = equal_error_rate([0.8, 0.9], [0.1, 0.4])
    assert res.threshold == pytest.approx(0.6)


def test_eer_matches_known_overlap():
    # One genuine below and one impostor above the crossing -> EER = 1/4.
    res = equal_error_rate([0.9, 0.8, 0.7, 0.35], [0.1, 0.2, 0.3, 0.75])
    assert res.eer == pytest.approx(0.25)


def test_tar_at_far_picks_lowest_admissible_threshold():
    genuine = [0.9, 0.8, 0.6, 0.5]
    impostor = [0.1, 0.2, 0.3, 0.7]
    op = tar_at_far(genuine, impostor, target_far=0.25)
    assert op.far <= 0.25
    assert op.tar == pytest.approx(1.0)  # threshold just above 0.3 keeps all genuine
    strict = tar_at_far(genuine, impostor, target_far=0.0)
    assert strict.far == 0.0 and strict.tar == pytest.approx(0.5)


def test_unresolvable_far_is_flagged():
    op = tar_at_far([0.9] * 10, [0.1] * 45, target_far=0.001)
    assert op.resolvable is False  # 1/45 = 2.2 % > 0.1 %


def test_far_resolution_for_45_impostors():
    r = far_resolution(45)
    assert r["smallest_nonzero"] == pytest.approx(1 / 45)
    assert r["rule_of_three_95"] == pytest.approx(3 / 45)
