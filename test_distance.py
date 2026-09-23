import unittest

from distance import parse_distance_furlongs


class TestDistance(unittest.TestCase):
    def test_supported_formats(self):
        examples = {
            "6f": 6.0,
            "1m": 8.0,
            "1m½f": 8.5,
            "1m2f": 10.0,
            "7½f": 7.5,
            "4m2½f": 34.5,
            " 2m ": 16.0,
        }

        for text, expected in examples.items():
            with self.subTest(distance=text):
                self.assertEqual(
                    parse_distance_furlongs(text), expected
                )

    def test_missing_values(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertIsNone(parse_distance_furlongs(value))

    def test_invalid_values(self):
        for value in ("f", "1mf", "0f", "1600m extra", "unknown"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_distance_furlongs(value)


if __name__ == "__main__":
    unittest.main()