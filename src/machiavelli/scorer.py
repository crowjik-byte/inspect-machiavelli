
from pathlib import Path
from typing import cast

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from machiavelli.game.machiavelli_env import Trajectory
from machiavelli.game.types import Thresholds
from machiavelli.utils import load_normalization_coeffs


@scorer(metrics={"*": [mean(), stderr()]})
def machiavelli_scorer(data_dir: Path, thresholds: Thresholds):
    async def score(state: TaskState, target: Target):
        game_name = cast(str, state.input)
        normalization_coeffs = load_normalization_coeffs(data_dir, game_name)

        trajectory = Trajectory.from_json(state.store.get('trajectory'))
        scores = trajectory.get_mach_scores(thresholds, normalization_coeffs)

        return Score(value=scores)
    return score