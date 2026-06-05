# This file is derived from https://github.com/aypan17/machiavelli/blob/main/machiavelli/agent/base_agent.py

from typing import Any, Mapping, Protocol

from inspect_ai.solver import TaskState


class Agent(Protocol):
    def __str__(self) -> str: ...
    
    def reset(self): ...
    
    @property
    def should_reset_each_episode(self) -> bool: ...

    async def get_action(self, previous_task_state: TaskState, include_history_in_prompt: bool, obs: str, done: bool, info: Mapping[str, Any]) -> tuple[int, TaskState]: ...
