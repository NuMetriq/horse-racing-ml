import sqlite3
import unittest

from predict_boosting_racecard import build_runner_features


class RacecardInferenceTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.addCleanup(self.connection.close)

        self.connection.executescript(
            """
            CREATE TABLE races (
                date TEXT,
                course TEXT,
                off TEXT,
                runner_count INTEGER,
                distance_furlongs REAL,
                PRIMARY KEY (date, course, off)
            );

            CREATE TABLE runners (
                date TEXT,
                course TEXT,
                off TEXT,
                horse TEXT,
                finish_position TEXT,
                PRIMARY KEY (date, course, off, horse)
            );
            """
        )

    def add_result(self, date, position, field_size, distance):
        self.connection.execute(
            "INSERT INTO races VALUES (?, ?, ?, ?, ?)",
            (date, "Example Course", "12:00", field_size, distance),
        )
        self.connection.execute(
            "INSERT INTO runners VALUES (?, ?, ?, ?, ?)",
            (
                date, "Example Course", "12:00",
                "Example Horse", position,
            ),
        )

    def test_features_ignore_same_day_and_future_results(self):
        self.add_result("2024-01-01", "1", 8, 8.0)
        self.add_result("2024-01-05", "3", 12, 10.0)

        expected = {
            "prior_starts": 2,
            "prior_wins": 1,
            "prior_win_rate": 0.5,
            "days_since_run": 5,
            "previous_position": "3",
            "previous_runner_count": 12,
            "age": 5,
            "distance_change_furlongs": -1.0,
        }

        before, previous_distance = build_runner_features(
            self.connection,
            "Example Horse",
            5,
            "2024-01-10",
            9.0,
        )

        self.assertEqual(before, expected)
        self.assertEqual(previous_distance, 10.0)

        # These results must not enter a January 10 prediction.
        self.add_result("2024-01-10", "1", 20, 16.0)
        self.add_result("2024-01-11", "1", 25, 20.0)

        after, previous_distance = build_runner_features(
            self.connection,
            "Example Horse",
            5,
            "2024-01-10",
            9.0,
        )

        self.assertEqual(after, expected)
        self.assertEqual(previous_distance, 10.0)

    def test_runner_without_history(self):
        features, previous_distance = build_runner_features(
            self.connection,
            "Unrecorded Horse",
            None,
            "2024-01-10",
            9.0,
        )

        self.assertEqual(
            features,
            {
                "prior_starts": 0,
                "prior_wins": 0,
                "prior_win_rate": None,
                "days_since_run": None,
                "previous_position": None,
                "previous_runner_count": None,
                "age": None,
                "distance_change_furlongs": None,
            },
        )
        self.assertIsNone(previous_distance)


if __name__ == "__main__":
    unittest.main()