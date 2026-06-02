from inspect_ai.solver import Generate, TaskState, solver


@solver
def machiavelli_solver():
    async def solve(state: TaskState, generate: Generate):
        # TODO: Implement
        return state
    return solve