import unittest

from prior_form import (
    calculate_prior_form,
    calculate_recent_form,
    calculate_days_since_run,
)


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

    def test_recent_form_window_and_same_day_exclusion(self):
        history = [
            ("2023-01-01", "Example Course", "1:00", "1"),
            ("2024-01-01", "Example Course", "1:00", "2"),
            ("2024-01-01", "Example Course", "3:00", "1"),
            ("2024-01-02", "Example Course", "1:00", "3"),
        ]

        features = calculate_recent_form(history, window_days=365)

        self.assertEqual(features[0][4:], (0, 0, None))
        self.assertEqual(features[1][4:], (1, 1, 1.0))
        self.assertEqual(features[2][4:], (1, 1, 1.0))
        self.assertEqual(features[3][4:], (2, 1, 0.5))

    def test_days_since_run_uses_previous_earlier_date(self):
        history = [
            ("2023-01-01", "Example Course", "1:00", "1"),
            ("2023-01-01", "Example Course", "3:00", "2"),
            ("2023-01-11", "Example Course", "1:00", "3"),
            ("2023-01-11", "Example Course", "3:00", "1"),
            ("2023-01-14", "Example Course", "1:00", "2"),
        ]

        gaps = calculate_days_since_run(history)

        self.assertEqual(gaps, [None, None, 10, 10, 3])


if __name__ == "__main__":
    unittest.main()