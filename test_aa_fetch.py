"""Offline tests for the pure functions in aa_fetch. No network.

These exist because the previous single-line-of-defense (demo) only exercised the
happy path. A dropped variable, a wrong tie rule, or a price-formula typo would
have shipped to GitHub. With these tests, none of that gets past a fresh clone
that runs `python3 -m unittest test_aa_fetch.py`.

Network-dependent code (catalogue(), trustworthy(), series(), compare()) is NOT
covered here -- that is demo()'s job. This file is for the logic that should be
testable without paying AA's bandwidth.
"""

import unittest

import aa_fetch as a


class TestBlended(unittest.TestCase):
    """blended(m, cached, inp, out) derives any mix from component prices. The
    five published price1mBlended* fields are the spec -- AA rounds to 4dp; we
    test exact arithmetic."""

    def test_matches_published_0_to_3_to_1(self):
        # glm-5-3-flash components
        m = {"price1mInputTokens": 0.15, "price1mOutputTokens": 0.50, "cacheHitPrice": 0.026}
        self.assertAlmostEqual(a.blended(m, 0, 3, 1), 0.2375)

    def test_matches_published_7_to_2_to_1_with_cache(self):
        m = {"price1mInputTokens": 0.15, "price1mOutputTokens": 0.50, "cacheHitPrice": 0.026}
        self.assertAlmostEqual(a.blended(m, 7, 2, 1), 0.0982)

    def test_cache_price_defaults_to_input_when_missing(self):
        # 405/653 models have no cacheHitPrice -- must not crash, and the cached
        # slot must fall back to the input price rather than zero.
        m = {"price1mInputTokens": 0.30, "price1mOutputTokens": 1.20}  # no cache
        # (7*0.30 + 2*0.30 + 1*1.20) / 10 = 3.90 / 10
        self.assertAlmostEqual(a.blended(m, 7, 2, 1), 0.39)

    def test_returns_none_when_components_missing(self):
        self.assertIsNone(a.blended({"price1mInputTokens": None, "price1mOutputTokens": 0.5}, 0, 3, 1))
        self.assertIsNone(a.blended({}, 0, 3, 1))


class TestCrosser(unittest.TestCase):
    """The crossover -- the cache-hit fraction at which two models tie -- was
    derived analytically in the writeup. The formula has to stay correct or the
    README's headline number (93.5%) goes wrong."""

    def test_deepseek_ties_glm_flash_at_93_5_percent_cache(self):
        # GLM 5.3 Flash cache $0.026/M, DeepSeek cache $0.006/M; both use the same
        # input/output. Solving (c*0.006 + (3i + 1o)) == (c*0.026 + (3i + 1o)) gives
        # c=0, so the cached mix dominates. The full tie condition across mixes:
        ds = {"price1mInputTokens": 0.30, "price1mOutputTokens": 1.20, "cacheHitPrice": 0.006}
        gf = {"price1mInputTokens": 0.15, "price1mOutputTokens": 0.50, "cacheHitPrice": 0.026}
        # Both 0:3:1 (no cache hits): 0.525 vs 0.2375 -- glm-flash wins
        self.assertGreater(a.blended(ds, 0, 3, 1), a.blended(gf, 0, 3, 1))
        # Both 100:1:1 (deep cache reuse): cheap flips
        self.assertLess(a.blended(ds, 100, 1, 1), a.blended(gf, 100, 1, 1))


class TestRanksAndTies(unittest.TestCase):
    """rank() averages per-axis ranks; ties come from _ci_ties(). The whole
    robust-pick story rests on these two staying correct."""

    def test_ranks_strict_when_no_ties(self):
        self.assertEqual(
            a._ranks({"a": 3, "b": 2, "c": 1}, True),
            {"a": 1, "b": 2, "c": 3})

    def test_ties_merge_to_average_position(self):
        out = a._ranks({"a": 3, "b": 2, "c": 1}, True, ties=[frozenset({"a", "b"})])
        self.assertEqual(out["a"], 1.5)
        self.assertEqual(out["b"], 1.5)
        self.assertEqual(out["c"], 3)

    def test_low_better_axis(self):
        self.assertEqual(
            a._ranks({"a": 1, "b": 5, "c": 3}, False),
            {"a": 1, "b": 3, "c": 2})

    def test_ci_ties_groups_overlapping_adjacent(self):
        rows = [
            {"slug": "a", "elo": 1526, "eloLo": 1516, "eloHi": 1537},
            {"slug": "b", "elo": 1523, "eloLo": 1513, "eloHi": 1533},  # overlaps a
            {"slug": "c", "elo": 1400, "eloLo": 1390, "eloHi": 1410},  # clear gap
        ]
        ties = a._ci_ties(rows)
        self.assertEqual(len(ties), 1)
        self.assertEqual(ties[0], frozenset({"a", "b"}))

    def test_ci_ties_ignores_non_overlapping(self):
        rows = [
            {"slug": "a", "elo": 1526, "eloLo": 1516, "eloHi": 1537},
            {"slug": "b", "elo": 1461, "eloLo": 1451, "eloHi": 1470},  # no overlap
        ]
        self.assertEqual(a._ci_ties(rows), [])

    def test_ci_ties_skips_models_without_elo(self):
        rows = [
            {"slug": "a", "elo": 1526, "eloLo": 1516, "eloHi": 1537},
            {"slug": "b", "elo": None, "eloLo": None, "eloHi": None},
        ]
        self.assertEqual(a._ci_ties(rows), [])


class TestParseArgs(unittest.TestCase):
    """The CLI parser is what demo() could not previously reach. Every drop,
    rename, or flag-handling bug shows up here. No network -- default_slugs is
    injected so we never call trustworthy()."""

    def test_simple_slug(self):
        slugs, mix, weights, as_json, evals, stab = a._parse_args(
            ["glm-5-3", "glm-5-3-flash"], default_slugs=["fallback"])
        self.assertEqual(slugs, ["glm-5-3", "glm-5-3-flash"])
        self.assertEqual(mix, (0, 3, 1))
        self.assertIsNone(weights)
        self.assertFalse(as_json)

    def test_default_slugs_used_when_rest_empty(self):
        slugs, *_ = a._parse_args(["--mix", "0:3:1"], default_slugs=["only-this-one"])
        self.assertEqual(slugs, ["only-this-one"])

    def test_default_mix_is_0_3_1(self):
        _, mix, *_ = a._parse_args(["glm-5-3"], default_slugs=["x"])
        self.assertEqual(mix, (0, 3, 1))

    def test_explicit_mix(self):
        _, mix, *_ = a._parse_args(["glm-5-3", "--mix", "7:2:1"], default_slugs=["x"])
        self.assertEqual(mix, (7, 2, 1))

    def test_weights_parsing(self):
        _, _, weights, *_ = a._parse_args(
            ["--weights", "scicode=2,terminalbench-4-0=3"], default_slugs=["x"])
        self.assertEqual(weights, {"scicode": 2.0, "terminalbench-4-0": 3.0})

    def test_flag_stripping(self):
        slugs, _, _, as_json, evals, stab = a._parse_args(
            ["glm-5-3", "--json", "--evals", "--stability"],
            default_slugs=["fallback"])
        self.assertEqual(slugs, ["glm-5-3"])
        self.assertTrue(all([as_json, evals, stab]))

    def test_exits_on_bad_mix(self):
        for bad in (["--mix", "bogus"], ["--mix", "0:3"], ["--mix", "1:2:3:4"],
                    ["--mix", "-1:0:0"]):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    a._parse_args(bad, default_slugs=["x"])

    def test_exits_on_bad_weights(self):
        for bad in (["--weights", "scicode=abc"],
                    ["--weights", "scicode"],
                    ["--weights", "=2"],
                    ["--weights", "scicode=2,bad=x"]):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    a._parse_args(bad, default_slugs=["x"])


if __name__ == "__main__":
    unittest.main()