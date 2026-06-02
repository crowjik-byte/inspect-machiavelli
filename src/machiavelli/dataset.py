from typing import Sequence

from inspect_ai.dataset import Sample

def machiavelli_dataset(games: list[str]) -> Sequence[Sample]:
    return [Sample(input=game) for game in games]
