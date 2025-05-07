from gymnasium import Wrapper
import numpy as np


class maze_runner_stickyActions(Wrapper):
    """CHANGE THIS!!!!"""
    def __init__(self, env, n_objects,sticky_prob = 0.25):
        super().__init__(env)

        self.sticky_prob = sticky_prob
        self.last_action = {}
        for k,v in n_objects.items():
            self.last_action[k] = [None for _ in range(v)]
        self.n_objects = n_objects
    def step(self, action):
        for k , v in self.n_objects.items():
            for a in range(v):
                
                if self.last_action[k][a] is not None and np.random.rand() < self.sticky_prob:
                    
                    action[k][a] = self.last_action[k][a]
                self.last_action[k][a] = action[k][a]
            
        
        return super().step(action)
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        for k,v in self.n_objects.items():
            self.last_action[k] = [None for _ in range(v)]

        return obs, info