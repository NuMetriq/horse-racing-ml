import math
import unittest

from feature_transforms import calculate_relative_finish, encode_age


class TestRelativeFinish(unittest.TestCase):
    def test_valid_positions(self):
        self.assertEqual(calculate_relative_finish("1", 6), 0.0)
        self.assertEqual(calculate_relative_finish("6", 6), 1.0)
        self.assertAlmostEqual(calculate_relative_finish("5", 6), 0.8)
        self.assertAlmostEqual(
            calculate_relative_finish("5", 20), 4 / 19
        )

    def test_unavailable_or_invalid_results(self):
        cases = [
            (None, 6),
            ("PU", 6),
            ("0", 6),
            ("8", 6),
            ("1", None),
            ("1", 1),
            ("1", 0),
            ("1", 6.5),
            ("1", float("nan")),
            ("1", float("inf")),
        ]

        for position, runner_count in cases:
            with self.subTest(
                position=position, runner_count=runner_count
            ):
                self.assertTrue(
                    math.isnan(
                        calculate_relative_finish(position, runner_count)
                    )
                )

    def test_valid_ages(self):
        self.assertEqual(encode_age(2), 2.0)
        self.assertEqual(encode_age(5), 5.0)
        self.assertEqual(encode_age("16"), 16.0)

    def test_missing_or_invalid_ages(self):
        for value in (
            None, "", "-", 0, 1, -1, 2.5,
            float("nan"), float("inf"),
        ):
            with self.subTest(value=value):
                self.assertTrue(math.isnan(encode_age(value)))


if __name__ == "__main__":
    unittest.main()