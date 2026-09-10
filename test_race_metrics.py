import math
import unittest

from race_metrics import evaluate_race_scores


class RaceMetricsTests(unittest.TestCase):
    def test_normalization_and_equal_race_weighting(self):
        scores = {
            ("2024-01-01", "Example A", "1:00"): [
                ("Horse A", "1", 3.0),
                ("Horse B", "2", 1.0),
            ],
            ("2024-01-02", "Example B", "2:00"): [
                ("Horse C", "1", 1.0),
                ("Horse D", "2", 1.0),
                ("Horse E", "3", 1.0),
            ],
        }

        model_loss, uniform_loss = evaluate_race_scores(scores)

        expected_model = (-math.log(0.75) - math.log(1 / 3)) / 2
        expected_uniform = (math.log(2) + math.log(3)) / 2

        self.assertAlmostEqual(model_loss, expected_model)
        self.assertAlmostEqual(uniform_loss, expected_uniform)


if __name__ == "__main__":
    unittest.main()