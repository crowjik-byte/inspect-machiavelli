import shutil
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

GAME_DATA_REPO = "kobyjl/machiavelli_game_data"
GAME_DATA_ARCHIVE = "game_data.zip"
GAME_DATA_PASSWORD = b"machiavelli"


def fetch_and_set_up_game_data(data_dir: Path) -> Path:
    game_data_dir = data_dir / 'game_data'
    if game_data_dir.exists():
        return game_data_dir

    latest_revision = HfApi().dataset_info(GAME_DATA_REPO).sha
    archive_path = hf_hub_download(
        repo_id=GAME_DATA_REPO,
        filename=GAME_DATA_ARCHIVE,
        repo_type="dataset",
        revision=latest_revision,
    )

    staging_dir = game_data_dir.parent / ".game_data.partial"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(path=staging_dir, pwd=GAME_DATA_PASSWORD)

    (staging_dir / "game_data").rename(game_data_dir)
    shutil.rmtree(staging_dir, ignore_errors=True)
    return game_data_dir
