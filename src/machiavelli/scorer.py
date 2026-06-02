
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState


@scorer(metrics={"*": [mean(), stderr()]})
def machiavelli_scorer():
    async def score(state: TaskState, target: Target):
        return Score(value=0.0) # TODO
    return score