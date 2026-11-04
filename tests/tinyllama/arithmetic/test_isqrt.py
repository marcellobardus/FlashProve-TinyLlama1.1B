import ref_independent as ref


def test_vectors_match_both_implementations():
    for case in ref.load_vectors("isqrt.json")["cases"]:
        a, expected = case["a"], case["expected"]
        assert ref.isqrt(a) == expected, case
        assert ref.isqrt_bisect(a) == expected, case


def test_definition_largest_r_with_r_squared_le_a():
    for case in ref.load_vectors("isqrt.json")["cases"]:
        a, r = case["a"], case["expected"]
        assert r * r <= a < (r + 1) * (r + 1), case


def test_bisection_agrees_with_math_isqrt_on_squares_and_neighbours():
    roots = [
        0, 1, 2, 3, 5, 12, 127, 128, 1000, 2047, 4095, 4096,
        2**13 + 7, 10**6, 2**31, 547599908735, 2**39 - 1, 2**39, 10**12,
    ]
    for r in roots:
        for a in (r * r - 1, r * r, r * r + 1) if r else (0,):
            assert ref.isqrt_bisect(a) == ref.isqrt(a), (a, r)


def test_bisection_agrees_on_vx_sums():
    for a in (
        2048 * 547599908735 ** 2,
        2048 * 2047 ** 2 + 85899,
        25 + 1,
        16760836 + 85899,
        2000000 + 1,
        (1 << 78) - 1,
        (1 << 78),
    ):
        assert ref.isqrt_bisect(a) == ref.isqrt(a), a
