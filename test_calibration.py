import math
import unittest

from calibrate_probabilities import adjust_probabilities


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.race_key = ("2024-01-01", "Example Course", "12:00")
        self.races = {
            self.race_key: [
                ("Horse A", "1", 0.6),
                ("Horse B", "0", 0.3),
                ("Horse C", "0", 0.1),
            ]
        }

    def test_gamma_one_preserves_predictions_and_labels(self):
        adjusted = adjust_probabilities(self.races, gamma=1.0)

        self.assertEqual(set(adjusted), set(self.races))

        for original, result in zip(
            self.races[self.race_key],
            adjusted[self.race_key],
            strict=True,
        ):
            self.assertEqual(original[:2], result[:2])
            self.assertAlmostEqual(original[2], result[2], places=14)

    def test_probabilities_sum_to_one_within_each_race(self):
        races = {
            **self.races,
            ("2024-01-01", "Other Course", "13:00"): [
                ("Horse D", "0", 0.8),
                ("Horse E", "1", 0.2),
            ],
        }

        adjusted = adjust_probabilities(races, gamma=1.237)

        for runners in adjusted.values():
            probabilities = [p for _, _, p in runners]

            self.assertTrue(all(0 < p < 1 for p in probabilities))
            self.assertAlmostEqual(
                math.fsum(probabilities), 1.0, places=14
            )

    def test_positive_gamma_preserves_rankings_and_ties(self):
        races = {
            self.race_key: [
                ("Horse A", "1", 0.4),
                ("Horse B", "0", 0.4),
                ("Horse C", "0", 0.2),
            ]
        }

        for gamma in (0.5, 1.0, 1.237, 2.0):
            with self.subTest(gamma=gamma):
                adjusted = adjust_probabilities(races, gamma)
                probabilities = {
                    horse: p
                    for horse, _, p in adjusted[self.race_key]
                }

                self.assertEqual(
                    probabilities["Horse A"],
                    probabilities["Horse B"],
                )
                self.assertGreater(
                    probabilities["Horse B"],
                    probabilities["Horse C"],
                )

    def test_gamma_two_matches_hand_calculation(self):
        adjusted = adjust_probabilities(self.races, gamma=2.0)

        # Squared probabilities: 0.36, 0.09, 0.01; total: 0.46.
        expected = (36 / 46, 9 / 46, 1 / 46)

        for runner, probability in zip(
            adjusted[self.race_key], expected, strict=True
        ):
            self.assertAlmostEqual(
                runner[2], probability, places=14
            )

    def test_invalid_gamma_is_rejected(self):
        for gamma in (
            0.0, -1.0, float("nan"),
            float("inf"), float("-inf"),
        ):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    adjust_probabilities(self.races, gamma)


if __name__ == "__main__":
    unittest.main()