from pathlib import Path

from inspect_ai import Task, task

from machiavelli.dataset import machiavelli_dataset
from machiavelli.game.types import Thresholds
from machiavelli.game_data import fetch_and_set_up_game_data
from machiavelli.scorer import machiavelli_scorer
from machiavelli.solver import machiavelli_solver

DEFAULT_DATA_DIR = Path('./outputs')

@task
def machiavelli(
    data_dir: Path = DEFAULT_DATA_DIR,
    games: list[str] | None = None,
    thresholds: Thresholds | None = None
):
    fetch_and_set_up_game_data(data_dir)
    thresholds = thresholds if thresholds is not None else Thresholds()

    return Task(
        dataset=machiavelli_dataset(games),
        solver=machiavelli_solver(data_dir),
        scorer=machiavelli_scorer(data_dir, thresholds),
    )