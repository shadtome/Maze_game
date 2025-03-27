from DQN.agents.basic import BaseAgent
import Maze_env.env
import Maze_env
from Maze_env.reward_functions.maze_runner import MazeRunnerRewardsFun
from Maze_env.wrappers.reward_wrappers.runner_rewards import MazeRunnerRewards
from Maze_env.game_info.maze_runner_info import MazeRunner


class MazeRunnerAgent(BaseAgent):
    
    def __init__(self, model, vision, action_type='full', dist_paradigm='radius', **kwargs):
        game_info = MazeRunner()
        game_info.rewards_dist = kwargs.pop('rewards_dist',game_info.rewards_dist)
        kwargs.pop('type_of_objects',None)
        kwargs.pop('collision_rules',None)
        kwargs.pop('objectives',None)
        kwargs.pop('maze_environment',None)
        
        super().__init__(model=model, vision=vision, action_type=action_type, 
                         dist_paradigm=dist_paradigm,game_info=game_info, **kwargs)
        
    @classmethod
    def load(cls, name, default_rewards = True):
        game_info = MazeRunner()
        default_rewards_dist = None
        if default_rewards == True:
            default_rewards_dist=game_info.rewards_dist

        return super().load(name = name, game_info = game_info,
                            rewards_cls=MazeRunnerRewardsFun,
                            default_rewards_dist=default_rewards_dist)

    def copy(self):
        return self.__class__(model = self.maze_model,
                              vision = self.vision,
                              action_type = self.action_type,
                              dist_paradigm = self.dist_paradigm)
    
    def add_wrappers(self, env):
        env = MazeRunnerRewards(env, rewards_dist = self.game_info.rewards_dist)
        return super().add_wrappers(env)
    
    def set_game(self, game_info):
        return super().set_game(game_info)
