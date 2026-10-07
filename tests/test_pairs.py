import pytest

from fpeval.pairs import expected_counts, genuine_pairs, impostor_pairs


def make_gallery(n_people, n_captures):
    return {p: [(p, c) for c in range(n_captures)] for p in range(n_people)}


@pytest.mark.parametrize("n_people,n_captures", [(10, 3), (20, 5), (2, 2)])
@pytest.mark.parametrize("mode", ["cross", "reference"])
def test_counts_match_closed_form(n_people, n_captures, mode):
    g = make_gallery(n_people, n_captures)
    expected = expected_counts(n_people, n_captures, mode)
    assert len(genuine_pairs(g)) == expected["genuine"]
    assert len(impostor_pairs(g, mode)) == expected["impostor"]


def test_10x3_reference_numbers():
    # 10 people x 3 captures: 10 * 3 = 30 genuine, 10*9/2 = 45 impostor.
    assert expected_counts(10, 3, "reference") == {"genuine": 30, "impostor": 45}
    assert expected_counts(10, 3, "cross") == {"genuine": 30, "impostor": 405}


def test_genuine_pairs_never_cross_people():
    for a, b in genuine_pairs(make_gallery(5, 4)):
        assert a[0] == b[0] and a != b


def test_impostor_pairs_always_cross_people():
    for a, b in impostor_pairs(make_gallery(5, 4), "cross"):
        assert a[0] != b[0]


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        impostor_pairs(make_gallery(3, 2), "all")
