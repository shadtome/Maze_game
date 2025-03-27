
from Maze_env.reward_functions.maze_runner import MazeRunnerRewardsFun
from Maze_env.game_info.basic_info import basicGame
from Maze_env.wrappers.reward_wrappers.runner_rewards import MazeRunnerRewards


class MazeRunner(basicGame):
    def __init__(self):

        super().__init__()
        rewards = MazeRunnerRewardsFun(
                        GOAL = 100.0,
                         SEE_GOAL = 0.00,
                         DONT_SEE_GOAL = -0.00,
                         NEW_PLACE = 50.0,
                         OLD_PLACE = -0.75,
                         GET_CLOSER = 50.0, 
                         GET_CLOSER_CONSTANT = 50.0,
                         GET_FARTHER = -30.0,
                         GET_FARTHER_CONSTANT = -30.00,
                         DIST = 0.0
        )
        self.name = 'Maze Runner'
        self.rewards_dist = {'agents': rewards}
        self.collision_rules = {}
        self.objectives = {'agents': 'goal'}
        self.type_of_objects = ['agents']
        self.maze_environment = 'Maze_env/MazeRunner-v0'
        self.reward_wrapper = MazeRunnerRewards
