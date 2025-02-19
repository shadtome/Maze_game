from gymnasium import Wrapper
from collections import deque
import numpy as np


# --- Reward Hyperparameters --- #
SEE_GOAL = 0.99
DONT_SEE_GOAL = -0.1
NEW_PLACE = 0.4
OLD_PLACE = -0.1
GET_CLOSER = 0.3
GET_FARTHER = -0.1

class maze_runner_rewards(Wrapper):
    def __init__(self,env):

        super().__init__(env)
        
        self.agents_dist = {}
        self.agents_recent_loc = {}
        self.agents_past = {}

        self.cum_rewards = []


    def step(self,action):
        new_obs, reward, terminated, truncated, info = super().step(action)
    
        for k in range(info['n_agents']):
            self.agents_dist[f'agent_{k}'].append(info[f'agent_{k}']['man_dist'])

            pos = info[f'agent_{k}']['pos']
            self.agents_recent_loc[f'agent_{k}'].append(pos)

            # Punish for going to the same place it has already visted
            if pos not in self.agents_past[f'agent_{k}']:
                self.agents_past[f'agent_{k}'].add(pos)
                
                reward[k] +=NEW_PLACE
            else:
               
                reward[k]+=OLD_PLACE

            # --- check neighborhoods for goals and other agents --- #
            index = -1
            for d in ['UP', 'DOWN','LEFT','RIGHT','UP_LEFT','UP_RIGHT','DOWN_LEFT','DOWN_RIGHT']:
                try:
                    index = info[f'agent_{k}'][f'{d}_vision'].index(3)
                except ValueError:
                    None
                
            if index!=-1:
                
                reward[k] += pow(SEE_GOAL,index+1)
            else:
               
                reward[k] += DONT_SEE_GOAL
            
            #if len(set(self.agents_recent_loc[f'agent_{k}']))<3:
            #    reward[k]-=0.2
            #else:
            #    reward[k]+=0.1

            if self.agents_dist[f'agent_{k}'][0]>self.agents_dist[f'agent_{k}'][1]:
                reward[k]+=GET_CLOSER
            else:
                reward[k]+=GET_FARTHER

            #if not info[f'agent_{k}']['done'] and truncated:
                #reward[k] -=info['timer']

            #reward[k] = np.clip(reward[k],-1,1)
            

            self.cum_rewards[k] += reward[k]
        
        #truncated = truncated or all(reward <-1*info['max_pos'] for reward in self.cum_rewards)
       
        return new_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        for a in range(info['n_agents']):
            self.agents_dist[f'agent_{a}'] = deque([info[f'agent_{a}']['man_dist']],maxlen=2)
            self.agents_recent_loc[f'agent_{a}'] = deque([info[f'agent_{a}']['pos']],maxlen=3)
            self.agents_past[f'agent_{a}'] = {info[f'agent_{a}']['pos']}
            self.cum_rewards.append(0)
        

        return obs, info