from Maze_env.reward_functions.maze_runner import MazeRunnerRewardsFun
from Maze_env.game_info.basic_info import basicGame
from Maze_env.wrappers.reward_wrappers.runner_rewards import MazeRunnerRewards
import matplotlib.cm as cm

class TeamMazeRunner(basicGame):
    def __init__(self,n_teams = 1):
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
        if n_teams>6:
            raise ValueError(f' Max number of teams exceeded at {n_teams}>10')
        self.n_teams = n_teams
        self.name = 'Team Maze Runner'
        self.rewards_dist = {}
        self.collision_rules = {}
        self.objectives = {}
        self.type_of_objects = []
        self.maze_environment = 'Maze_env/MazeRunner-v0'
        self.reward_wrapper = [MazeRunnerRewards]
        self.colormap = {}
        colormap_names = ["Blues", "Reds", "Greens", "Purples", "Oranges","Greys"]
        for i in range(n_teams):
            self.rewards_dist[f'team_{i}'] = rewards
            self.type_of_objects.append(f'team_{i}')
            self.objectives[f'team_{i}'] = 'goal'
            self.colormap[f'team_{i}'] = colormap_names[i]
            