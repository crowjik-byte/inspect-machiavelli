# This file is derived from https://github.com/aypan17/machiavelli/blob/main/machiavelli/agent/base_agent.py

from typing import Any, Mapping, Protocol

from inspect_ai.solver import TaskState


class Agent(Protocol):
    def __str__(self) -> str: ...
    
    def reset(self): ...
    
    @property
    def should_reset_each_episode(self) -> bool: ...

    async def get_action(self, raw_task_state: TaskState, obs: str, done: bool, info: Mapping[str, Any]) -> int: ... # TODO: better info type
