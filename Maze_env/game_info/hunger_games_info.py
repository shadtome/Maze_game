from Maze_env.reward_functions.hunger_games import HungerGamesRewardsFun
from Maze_env.game_info.basic_info import basicGame
from Maze_env.wrappers.reward_wrappers.hunger_games_rewards import HungerGamesRewards


class HungerGames(basicGame):
    def __init__(self):
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
                )
        self.name = 'Hunger Games'
        self.rewards_dist = {'agents':rewards}
        self.type_of_objects = ['agents']
        self.collision_rules = {('agents','agents'): {'agents': 'die', 'agents':'die'}}
        self.objectives = {'agents': 'goal'}
        self.maze_environment = 'Maze_env/MazeRunner-v0'
        self.reward_wrapper = HungerGamesRewards