
from DQN.agents.runner_agent import MazeRunnerAgent
from DQN.agents.basic import BaseAgent
from Maze_env.reward_functions.hunger_games import HungerGamesRewardsFun
from Maze_env.wrappers.reward_wrappers.hunger_games_rewards import HungerGamesRewards
from Maze_env.game_info.hunger_games_info import HungerGames



class HungerGamesAgent(BaseAgent):
    
    def __init__(self, model, 
                 vision, action_type='full', dist_paradigm='radius', **kwargs):
        game_info = HungerGames()
        game_info.rewards_dist = kwargs.pop('rewards_dist',game_info.rewards_dist)
        kwargs.pop('type_of_objects',None)
        kwargs.pop('collision_rules',None)
        kwargs.pop('objectives',None)
        kwargs.pop('maze_environment',None)
        
        super().__init__(model=model, vision=vision, action_type=action_type, 
                         dist_paradigm=dist_paradigm, 
                         game_info = game_info **kwargs)
        
    
    @classmethod
    def load(cls, name, default_rewards = True):
        game_info = HungerGames()
        default_rewards_dist = None
        if default_rewards == True:
            default_rewards_dist = game_info.rewards_dist
        return super().load(name = name,game_info = game_info,
                            rewards_cls=HungerGamesRewardsFun,
                            default_rewards_dist=default_rewards_dist
                            )
    
    def copy(self):
        return self.__class__(model = self.maze_model,
                              vision = self.vision,
                              action_type = self.action_type,
                              dist_paradigm = self.dist_paradigm)
    
    def add_wrappers(self, env):
        env = HungerGamesRewards(env, rewards_dist = self.game_info.rewards_dist)
        return super().add_wrappers(env)
    
    def freeze_base(self,freeze=True):
        self.Q_fun['agents'].freeze_base(freeze)

    def set_game(self, game_info):
        return super().set_game(game_info)