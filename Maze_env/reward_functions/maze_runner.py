from Maze_env.reward_functions.basic import BasicRewardFun


class MazeRunnerRewardsFun(BasicRewardFun):
    def __init__(self,**kwargs):

        super().__init__(**kwargs)

    def __setup__(self):
        rewards = {
            'SEE_GOAL': 0.0,
            'DONT_SEE_GOAL': -0.0,
            'NEW_PLACE': 0.0,
            'OLD_PLACE': -0.0,
            'GET_CLOSER': 0.0,
            'GET_CLOSER_CONSTANT': 0.0,
            'GET_FARTHER': -0.0,
            'GET_FARTHER_CONSTANT': -0.0,
            'DIST': 0.0,
        }
        return rewards | super().__setup__()