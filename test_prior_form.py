import unittest

from prior_form import (
    calculate_prior_form,
    calculate_recent_form,
    calculate_days_since_run,
    calculate_previous_position,
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

    def test_previous_position_excludes_same_day_results(self):
        history = [
            ("2023-01-01", "Course A", "1:00", "5"),
            ("2023-01-10", "Course A", "2:00", "2"),
            ("2023-01-10", "Course B", "3:00", "3"),
            ("2023-01-20", "Course A", "1:00", "PU"),
            ("2023-02-01", "Course A", "1:00", "4"),
            ("2023-02-10", "Course A", "1:00", "1"),
        ]

        actual = calculate_previous_position(history)

        self.assertEqual(
            actual,
            [None, "5", "5", None, "PU", "4"],
        )

if __name__ == "__main__":
    unittest.main()