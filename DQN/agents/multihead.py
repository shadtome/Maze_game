from DQN.agents.basic import BaseAgent
from DQN.agents.basic import baseline_rw
import numpy as np
from DQN.models.base import MultiHead
from typing import Literal, Optional


class MultiHeadAgent(BaseAgent):
    def __init__(self,maze_model,vision,
                 action_type = 'full',
                 rewards_dist = baseline_rw,
                 dist_paradigm = 'radius',
                 n_heads = 1,
                 head_function= None,
                   **kwargs):
        super().__init__(maze_model,vision,action_type,rewards_dist,dist_paradigm,n_heads=n_heads,**kwargs)

        # -- check if the inputs are correct -- #
        self.n_heads = n_heads
        self.head_function = head_function



    def __init_model__(self, maze_model,CNN_shape,n_actions, n_heads, **kwargs):
        if maze_model != MultiHead:
            raise ValueError('NN model should be a MultiHead class')
        return maze_model(CNN_shape,n_actions,n_heads)
    
    def cur_dist(self,level):
        if self.head_function == None:
            return None
        else:
             key = next(k for k,v in self.head_function.items() if v==level)
             return key.stop - 1
        

    def get_head(self,dist):
        # do it based on the max number of heads and correlate it with 
        # the number of heads
        if self.head_function == None:
             return dist - 1
        # Now do it based on the distribution
        for k,v in self.head_function.items():
             if dist in k:
                  return v
        print(f'Went through head function and did not find range')
        print(f'dist: {dist}\n head function: {self.head_function}')
        return 0
        

    def get_single_agent_action(self,env,state,a,info,epsilon):
        if np.random.random()<epsilon:
            action = int(env.action_space.sample())
        else:
            
            local_state_tensor = self.transform_local_to_nn(state[f'local_{a}'])
            global_state_tensor = self.transform_global_to_nn(state[f'global_{a}'])
            head = self.get_head(info[f'agent_{a}']['dist'])
            q_values = self.Q_fun(local_state_tensor,global_state_tensor,head)
            action=int(q_values.argmax().item())
        return action
    

    def __getModelparam__(self): 
            model_param = super().__getModelparam__()
            model_param['n_heads'] = self.n_heads    
            return model_param