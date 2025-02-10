from gymnasium import Wrapper
from collections import deque

class maze_runner_rewards(Wrapper):
    def __init__(self,env):
        super().__init__(env)
        
        self.agents_dist = {}
        self.agents_recent_loc = {}
        
        

    def step(self,action):

        new_obs, reward, terminated, truncated, info = self.env.step(action)
    
        for k in range(len(reward)):
            self.agents_dist[f'agent_{k}'].append(info[f'agent_{k}']['man_dist'])
            self.agents_recent_loc[f'agent_{k}'].append(info[f'agent_{k}']['pos'])

            if len(set(self.agents_recent_loc[f'agent_{k}']))<2:
                reward[k]-=1
            else:
                reward[k]+=1

            if self.agents_dist[f'agent_{k}'][0]>self.agents_dist[f'agent_{k}'][1]:
                reward[k]+=1

            if not info[f'agent_{k}']['done'] and truncated:
                reward[k] -=100
            

        


        
        
        

        return new_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        for a in range(len(obs)):
            self.agents_dist[f'agent_{a}'] = deque([info[f'agent_{a}']['man_dist']],maxlen=2)
            self.agents_recent_loc[f'agent_{a}'] = deque([info[f'agent_{a}']['pos']],maxlen=3)

        

        return obs, info