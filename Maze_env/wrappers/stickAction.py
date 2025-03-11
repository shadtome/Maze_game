from gymnasium import Wrapper
import numpy as np


class maze_runner_stickyActions(Wrapper):
    def __init__(self, env, n_agents,sticky_prob = 0.25):
        super().__init__(env)

        self.sticky_prob = sticky_prob
        self.last_action = [None]*n_agents
        self.n_agents = n_agents
    def step(self, action):
        for a in range(len(action)):
            
            if self.last_action[a] is not None and np.random.rand() < self.sticky_prob:
                action[a] = self.last_action[a]
            self.last_action[a] = action[a]
            
        
        return super().step(action)
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        self.last_action = [None]*self.n_agents

        return obs, info