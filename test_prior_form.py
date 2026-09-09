import unittest

from prior_form import calculate_prior_form


class PriorFormTests(unittest.TestCase):
    def test_same_day_results_are_excluded(self):
        history = [
            ("2023-01-01", "Example Course", "1:00", "1"),
            ("2023-01-01", "Example Course", "3:00", "2"),
            ("2023-01-02", "Example Course", "1:00", "3"),
        ]

        features = calculate_prior_form(history)

        self.assertEqual(features[0][4:], (0, 0, None))
        self.assertEqual(features[1][4:], (0, 0, None))
        self.assertEqual(features[2][4:], (2, 1, 0.5))


if __name__ == "__main__":
    unittest.main()