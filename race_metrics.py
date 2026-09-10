import math


def evaluate_race_scores(race_scores):
    model_losses = []
    uniform_losses = []

    for race_key, runners in race_scores.items():
        total_score = sum(score for _, _, score in runners)

        winner_scores = [
            score
            for _, position, score in runners
            if position == "1"
        ]

        if len(winner_scores) != 1:
            raise ValueError(f"Expected one winner: {race_key}")

        winner_probability = winner_scores[0] / total_score

        model_losses.append(-math.log(winner_probability))
        uniform_losses.append(math.log(len(runners)))

    model_loss = sum(model_losses) / len(model_losses)
    uniform_loss = sum(uniform_losses) / len(uniform_losses)

    return model_loss, uniform_loss