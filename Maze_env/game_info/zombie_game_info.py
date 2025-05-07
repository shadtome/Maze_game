from Maze_env.reward_functions.hunger_games import HungerGamesRewardsFun
from Maze_env.game_info.basic_info import basicGame
from Maze_env.wrappers.reward_wrappers.hunger_games_rewards import HungerGamesRewards


class ZombieGame(basicGame):
    def __init__(self):
        zombie_rewards = None
        leon_rewards = None

        self.name = 'Zombie Games'
        self.rewards_dist = {'leon':leon_rewards,'zombie':zombie_rewards}
        self.type_of_objects = ['leon','zombie']
        self.collision_rules = {('leon','zombie'): {'leon': 'die', 'agents':'die'}}
        self.objectives = {'leon': 'goal','zombie':'kill'}
        self.maze_environment = 'Maze_env/MazeRunner-v0'
        self.reward_wrapper = [HungerGamesRewards,HungerGamesRewards]
        self.colormap = {'leon':'tab10','zombie':'Purples'}