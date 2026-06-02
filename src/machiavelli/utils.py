# This file is partially adapted from https://github.com/aypan17/machiavelli/blob/main/machiavelli/utils.py

import json
from pathlib import Path
import random
from re import Match
import numpy as np
from pydantic import TypeAdapter

from machiavelli.game.types import (
    GameAnnotations,
    GameMetadata,
    GameSource,
    GameSourceLine,
    GameTreeNode,
    GameVariableAnnotations,
    NormalizationStats,
)

def random_seed(seed: int) -> None:
    """
    Manually set random seed for various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_conditional_group(lines: list[GameSourceLine]):
    """
    returns all lines that appear after *if.
    stop when encounting anything other than *elseif or *else
    """
    assert lines[0][0].startswith("*if")
    wspace = lines[0][1]
    i = 1
    while i < len(lines):
        if lines[i][1] > wspace:
            i += 1
        elif lines[i][1] == wspace:
            if lines[i][0].startswith(("*elseif", "*elsif", "*else")):
                i += 1
            else:
                break
        else:
            break
    return lines[:i], lines[i:]


def get_nested_group(lines: list[GameSourceLine]):
    """
    returns the first lines and any consecutive lines with deeper nesting than it
    """
    assert len(lines) > 0
    wspace = lines[0][1]
    i = 1
    while i < len(lines) and (lines[i][1] > wspace):
        i += 1
    return lines[:i], lines[i:]


def make_lowercase(m: Match[str]) -> str:
    if m.group(1) is None:
        return m.group(0)
    return m.group(1).lower()


def make_underscore(m: Match[str]) -> str:
    if m.group(1) is None:
        return m.group(0)
    return m.group(1).lower()+"_"


def split_into_params(text: str) -> list[str]:
    """
    split text into params. this is wrong if text contains any nested quotes
    """
    split = []
    parens = 0
    in_quote = False
    curr = ""
    for c in text:
        if c.isspace():
            if parens == 0 and not in_quote:
                split.append(curr)
                curr = ""
                continue
        elif c == '(':
            parens += 1
        elif c == ')':
            parens -= 1
        elif c == '"':
            in_quote = not in_quote
        curr += c
    split.append(curr)
    split_ = []
    for s in split:
        if s.strip() != '':
            split_.append(s.strip())
    return split_


_game_metadata_adapter = TypeAdapter(GameMetadata)
_game_tree_adapter = TypeAdapter(dict[str, GameTreeNode])
_game_variable_annotations_adapter = TypeAdapter(GameVariableAnnotations)
_normalization_stats_adapter = TypeAdapter(dict[str, NormalizationStats])


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_game_metadata(data_dir: Path, game_name: str) -> GameMetadata:
    all_metadata = _load_json(data_dir / "game_data" / "game_metadata.json")
    return _game_metadata_adapter.validate_python(all_metadata[game_name])


def load_game_tree(data_dir: Path, game_name: str) -> dict[str, GameTreeNode]:
    raw = _load_json(data_dir / "game_data" / "trees" / f"{game_name}_game_tree.json")
    return _game_tree_adapter.validate_python(raw)


def load_game_annotations(data_dir: Path, game_name: str) -> GameAnnotations:
    return GameAnnotations.model_validate(
        _load_json(data_dir / "game_data" / "annotations_clean" / f"{game_name}.json")
    )


def load_game_variable_annotations(data_dir: Path, game_name: str) -> GameVariableAnnotations:
    all_variable_annotations = _load_json(data_dir / "game_data" / "annotations_clean" / "variables.json")
    return _game_variable_annotations_adapter.validate_python(all_variable_annotations[game_name])


def load_game_source(data_dir: Path, game_name: str) -> GameSource:
    source = GameSource.model_validate(_load_json(data_dir / "game_data" / "source" / f"{game_name}.json"))
    # Append the implicit end-of-scene transition ChoiceScript performs when
    # execution falls off the end of a scene, plus a buffer line so _inc_idx's
    # lines[idx+1] lookahead stays in range on the final command.
    for scene, lines in source.scenes.items():
        lines.append(("*finish" if scene != source.scene_list[-1] else "*ending", 0))
        lines.append(("", 0))
    return source


def load_normalization_coeffs(data_dir: Path, game_name: str) -> dict[str, NormalizationStats]:
    all_coeffs = _load_json(data_dir / "game_data" / "normalization_coeffs.json")
    return _normalization_stats_adapter.validate_python(all_coeffs[game_name])
