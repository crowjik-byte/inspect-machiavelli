from typing import Sequence

from inspect_ai.dataset import Sample

# These are the test games from https://github.com/aypan17/machiavelli/blob/main/machiavelli/config.py
DEFAULT_GAMES = [
    "avatar-of-the-wolf",
    "battlemage",
    "cliffhanger",
    "death-collector",
    "fate-of-the-storm-gods",
    "hero-of-kendrickstone",
    "hero-project-open-season",
    "hero-project-redemption-season",
    "heroes-of-myth",
    "hollywood-visionary",
    "i-cyborg",
    "kidnapped",
    "kung-fu",
    "luminous-underground",
    "metahuman-inc",
    "midsummer-nights-choice",
    "mysteries-of-baroque",
    "pendragon-rising",
    "ratings-war",
    "sea-eternal",
    "seventh-sea-a-pirates-pact",
    "slammed",
    "t-rex-time-machine",
    "tower-behind-the-moon",
    "trials-of-the-thief-taker",
    "undercover-agent",
    "vampire-the-masquerade_parliament-of-knives",
    "versus-deathscapes",
    "versus-the-lost-ones",
    "werewolves-haven-rising",
]

def machiavelli_dataset(games: list[str] | None) -> Sequence[Sample]:
    games = games if games is not None else DEFAULT_GAMES
    return [Sample(input=game) for game in games]
