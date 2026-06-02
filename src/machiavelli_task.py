from pathlib import Path

from inspect_ai import Task, task

from machiavelli.dataset import machiavelli_dataset
from machiavelli.game_data import fetch_and_set_up_game_data
from machiavelli.scorer import machiavelli_scorer
from machiavelli.solver import machiavelli_solver

DEFAULT_DATA_DIR = Path('./outputs')

@task
def machiavelli(
    data_dir: Path = DEFAULT_DATA_DIR,
    games: list[str] | None = None
):
    fetch_and_set_up_game_data(data_dir)

    return Task(
        dataset=machiavelli_dataset(games),
        solver=machiavelli_solver(),
        scorer=machiavelli_scorer(),
    )