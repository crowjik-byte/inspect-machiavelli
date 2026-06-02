from pathlib import Path
from typing import cast

from inspect_ai.solver import Generate, TaskState, solver

from machiavelli.agent.agent import Agent
from machiavelli.agent.inspect_agent import InspectAgent
from machiavelli.game.machiavelli_env import MachiavelliEnv, Trajectory
from machiavelli.utils import load_game_annotations, load_game_metadata, load_game_source, load_game_tree, load_game_variable_annotations


@solver
def machiavelli_solver(data_dir: Path, max_traj_length: int):
    async def solve(state: TaskState, generate: Generate):
        game_name = cast(str, state.input)

        agent = InspectAgent(generate)
        env = MachiavelliEnv(
            game_name,
            load_game_metadata(data_dir, game_name),
            load_game_tree(data_dir, game_name),
            load_game_annotations(data_dir, game_name),
            load_game_variable_annotations(data_dir, game_name),
            load_game_source(data_dir, game_name),
        )

        trajectory = await _run_trajectory(agent, env, data_dir, state, state.epoch, max_traj_length)

        state.store.set('trajectory', trajectory.to_json())

        return state
    return solve

async def _run_trajectory(agent: Agent, env: MachiavelliEnv, data_dir: Path, raw_task_state: TaskState, seed: int, max_traj_length) -> Trajectory:
    agent.reset()
    obs, info = env.reset(seed=seed)

    done = False
    step_count = 0
    while not done:
        action = await agent.get_action(raw_task_state, obs, done, info)
        obs, done, info = env.step(action)

        step_count += 1
        # Terminate early if we've reached the maximum trajectory length
        if step_count >= max_traj_length:
            print("Reached maximum trajectory length! Terminating early.")
            done = True

    return env.trajectory