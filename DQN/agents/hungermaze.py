
from DQN.agents.runner_agent import MazeRunnerAgent
from DQN.agents.basic import BaseAgent
from Maze_env.reward_functions.hunger_games import HungerGamesRewardsFun
from Maze_env.wrappers.reward_wrappers.hunger_games_rewards import HungerGamesRewards

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
rewards_dist = {'agents':rewards}
type_of_objects = ['agents']
collision_rules = {('agents','agents'): {'agents': 'die', 'agents':'die'}}
objectives = {'agents': 'goal'}
maze_environment = 'Maze_env/MazeRunner-v0'

class HungerGamesAgent(BaseAgent):
    def __init__(self, model, 
                 vision, action_type='full', dist_paradigm='radius', **kwargs):
        
        super().__init__(model=model, vision=vision, action_type=action_type, 
                         rewards_dist=rewards_dist, 
                         dist_paradigm=dist_paradigm, 
                         type_of_objects=type_of_objects, 
                         collision_rules=collision_rules,
                          objectives=objectives,
                           maze_environment=maze_environment, **kwargs)
        
    
    @classmethod
    def load(cls, name,):

        return super().load(name = name, 
                            type_of_objects=type_of_objects, 
                            collision_rules=collision_rules,
                            objectives=objectives,
                            maze_environment=maze_environment,
                            rewards_cls=HungerGamesRewardsFun)
    
    def copy(self):
        return self.__class__(model = self.maze_model,
                              vision = self.vision,
                              action_type = self.action_type,
                              dist_paradigm = self.dist_paradigm)
    
    def add_wrappers(self, env):
        env = HungerGamesRewards(env, rewards_dist = self.rewards_dist)
        return super().add_wrappers(env)