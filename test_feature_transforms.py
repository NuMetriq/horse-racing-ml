import math
import unittest

from feature_transforms import calculate_relative_finish


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


if __name__ == "__main__":
    unittest.main()