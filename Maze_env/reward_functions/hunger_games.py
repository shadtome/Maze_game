from Maze_env.reward_functions.maze_runner import MazeRunnerRewardsFun


class HungerGamesRewardsFun(MazeRunnerRewardsFun):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __setup__(self):
        rewards = {
            'HIT_OTHER': -1.0,
            'TOO_CLOSE': -0.0,
            'TOO_CLOSE_CONSTANT': -0.0
        }
        return rewards | super().__setup__()