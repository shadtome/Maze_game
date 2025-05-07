from DQN.agents.basic import BaseAgent
from DQN.agents.basic import baseline_rw
import numpy as np
from DQN.models.base import MultiHead
from typing import Literal, Optional
from Maze_env.game_info.basic_info import basicGame



class MultiHeadAgent(BaseAgent):
    def __init__(self, model, 
                 vision, action_type='full', 
                  dist_paradigm='radius', 
                game_info = basicGame, **kwargs):
        
        #for obj_type in model:
        #    if model[obj_type] != MultiHead or issubclass(model[obj_type],MultiHead)==False:
        #        raise ValueError(f'Model for {obj_type} must be MultiHead')
        super().__init__(model, vision, action_type, 
                          dist_paradigm,game_info = game_info, **kwargs)
        self.change_Q_net_heads(self.game_info)


    def change_Q_net_heads(self,game_info):
        for obj_type in game_info.type_of_objects:
            self.Q_fun[obj_type].set_game(game_info.name)


    def add_wrappers(self, env):
        env = self.game_info.add_wrapper(env)
        return super().add_wrappers(env)

    def set_game(self, game_info):
        self.change_Q_net_heads(game_info)
        return super().set_game(game_info)
        
