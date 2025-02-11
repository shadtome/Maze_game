from gymnasium import Wrapper
from collections import deque
import numpy as np

class maze_runner_rewards(Wrapper):
    def __init__(self,env,gamma = 0.99,epsilon = 1e-8,):

        super().__init__(env)
        
        self.agents_dist = {}
        self.agents_recent_loc = {}

        self.reward_mean = 0
        self.reward_var = 1
        self.reward_count = 0
        self.reward_decay = 0.99
        
    def normalize_rewards(self,rewards):
        """Normalize rewards using running statistics."""
        # Update running mean and variance
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        # Exponentially decay the moving average and variance
        self.reward_mean = self.reward_decay * self.reward_mean + (1 - self.reward_decay) * batch_mean
        self.reward_var = self.reward_decay * self.reward_var + (1 - self.reward_decay) * batch_var
        self.reward_count += 1
        
        # Normalize rewards
        normalized_rewards = (rewards - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)  # Avoid division by zero
        return normalized_rewards


    def step(self,action):
        new_obs, reward, terminated, truncated, info = super().step(action)
    
        for k in range(len(reward)):
            self.agents_dist[f'agent_{k}'].append(info[f'agent_{k}']['man_dist'])
            self.agents_recent_loc[f'agent_{k}'].append(info[f'agent_{k}']['pos'])

            if len(set(self.agents_recent_loc[f'agent_{k}']))<2:
                reward[k]-=0.2
            else:
                reward[k]+=0.1

            #if self.agents_dist[f'agent_{k}'][0]>self.agents_dist[f'agent_{k}'][1]:
            #    reward[k]+=1
            #else:
            #    reward[k]-=0.5

            if not info[f'agent_{k}']['done'] and truncated:
                reward[k] -=info['timer']
        
        #reward = self.normalize_rewards(reward)
        

        return new_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        for a in range(info['n_agents']):
            self.agents_dist[f'agent_{a}'] = deque([info[f'agent_{a}']['man_dist']],maxlen=2)
            self.agents_recent_loc[f'agent_{a}'] = deque([info[f'agent_{a}']['pos']],maxlen=3)

        

        return obs, info