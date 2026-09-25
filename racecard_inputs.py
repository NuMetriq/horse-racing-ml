import csv


def load_racecard(path):
    """Read exact horse names and current ages from a racecard CSV."""
    runners = []
    seen_horses = set()

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != ["horse", "age"]:
            raise ValueError(
                "Racecard CSV must have exactly these columns: horse,age"
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"Malformed CSV row at line {line_number}"
                )

            horse = row["horse"].strip()
            age_text = row["age"].strip()

            if not horse:
                raise ValueError(
                    f"Blank horse name at line {line_number}"
                )

            if horse in seen_horses:
                raise ValueError(f"Duplicate horse name: {horse}")

            if age_text == "":
                age = None
            else:
                try:
                    age = int(age_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Age must be an integer or blank for {horse}"
                    ) from exc

                if age < 1:
                    raise ValueError(
                        f"Age must be positive for {horse}"
                    )

            seen_horses.add(horse)
            runners.append({"horse": horse, "age": age})

    if len(runners) < 2:
        raise ValueError("Supply at least two runners")

    return runners