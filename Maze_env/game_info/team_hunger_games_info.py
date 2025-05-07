from Maze_env.reward_functions.hunger_games import HungerGamesRewardsFun
from Maze_env.game_info.basic_info import basicGame
from Maze_env.wrappers.reward_wrappers.hunger_games_rewards import HungerGamesRewards
import matplotlib.cm as cm

class TeamHungerGames(basicGame):
    def __init__(self,n_teams = 1):
        rewards = HungerGamesRewardsFun(
                        GOAL = 100.0,
                         SEE_GOAL = 0.00,
                         DONT_SEE_GOAL = -0.00,
                         NEW_PLACE = 50.0,
                         OLD_PLACE = -0.75,
                         GET_CLOSER = 50.0, 
                         GET_CLOSER_CONSTANT = 50.0,
                         GET_FARTHER = -30.0,
                         GET_FARTHER_CONSTANT = -30.00,
                         DIST = 0.0,
                         HIT_OTHER = -50.0,
                         TOO_CLOSE = -25.0,
                         TOO_CLOSE_CONSTANT = -25.0,
                         NO_NEIGHBORS = 1.0
                )
        if n_teams>6:
            raise ValueError(f' Max number of teams exceeded at {n_teams}>10')
        self.n_teams = n_teams
        self.name = 'Team Hunger Games'
        self.rewards_dist = {}
        self.type_of_objects = []
        self.collision_rules = {}
        self.objectives ={}
        self.maze_environment = 'Maze_env/MazeRunner-v0'
        self.reward_wrapper = [HungerGamesRewards]
        self.colormap = {}
        colormap_names = ["Blues", "Reds", "Greens", "Purples", "Oranges","Greys"]
        for i in range(n_teams):
            self.rewards_dist[f'team_{i}'] = rewards
            self.type_of_objects.append(f'team_{i}')
            self.objectives[f'team_{i}'] = 'goal'
            self.colormap[f'team_{i}'] = colormap_names[i]
            for j in range(n_teams):
                if j!=i:
                    self.collision_rules[(f'team_{i}',f'team_{j}')] = {f'team_{i}':'die',f'team_{j}':'die'}

