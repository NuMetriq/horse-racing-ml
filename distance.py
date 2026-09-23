import re


DISTANCE_PATTERN = re.compile(
    r"(?:(?P<miles>[0-9]+)m)?"
    r"(?:(?P<furlongs>[0-9]*)(?P<half>½)?f)?"
)


def parse_distance_furlongs(value: str | None) -> float | None:
    """Convert a recorded distance to furlongs.

    Missing values return None. Unsupported formats raise ValueError.
    """
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    match = DISTANCE_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"Unsupported distance: {value!r}")

    miles = match.group("miles")
    furlongs = match.group("furlongs")
    half = match.group("half")

    # Reject a bare 'f', including forms such as '1mf'.
    if furlongs == "" and half is None:
        raise ValueError(f"Unsupported distance: {value!r}")

    result = (
        8 * int(miles or 0)
        + int(furlongs or 0)
        + (0.5 if half else 0.0)
    )

    if result <= 0:
        raise ValueError(f"Distance must be positive: {value!r}")

    return result