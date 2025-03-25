from DQN.agents.basic import BaseAgent
from Maze_env.reward_functions.hunger_games import HungerGamesRewardsFun

runner_rewards = HungerGamesRewardsFun(
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

monster_rewards = None

class OutlastAgent(BaseAgent):
    def __init__(self,):
        None