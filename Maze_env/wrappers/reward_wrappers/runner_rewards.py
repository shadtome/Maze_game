from gymnasium import Wrapper
from collections import deque
import numpy as np


    
class MazeRunnerRewards(Wrapper):
    def __init__(self,env, rewards_dist):

        super().__init__(env)

        # -- rewards distribution -- #
        self.rewards_dist = rewards_dist
        
        self.objects_dist = {}
        self.objects_recent_loc = {}
        self.objects_past = {}

        self.cum_rewards = {}


    def step(self,action):
        new_obs, reward, terminated, truncated, info = super().step(action)

        for object in info['type_of_objects']:
            if 'goal' in info[object]:
                for k in range(info['n_'+object]):
                    self.objects_dist[object + f'_{k}'].append(info[object + f'_{k}']['dist'])

                    # --- positions of the agent and the goal --- #
                    pos = info[object + f'_{k}']['pos']
                    t_pos = info[object + f'_{k}']['goal']
                    agent_done = info[object + f'_{k}']['done']
                    self.objects_recent_loc[object + f'_{k}'].append(pos)

                    # --- punish for revisiting spots --- #
                    if pos not in self.objects_past[object + f'_{k}']:
                        self.objects_past[object + f'_{k}'][pos] = 1
                        
                        reward[object][k] +=self.rewards_dist[object]['NEW_PLACE']/(1 + pow(info[object+f'_{k}']['dist'],1))
                    else:
                        # --- punish corresponding to how many times it visited --- #
                        self.objects_past[object + f'_{k}'][pos]+=1
                        reward[object][k]+=self.rewards_dist[object]['OLD_PLACE']* float(self.objects_past[object + f'_{k}'][pos])

                    # --- check neighborhoods for goals  --- #
                    index_goal = -1
                    for d in ['CENTER','UP', 'DOWN','LEFT','RIGHT','UP_LEFT','UP_RIGHT','DOWN_LEFT','DOWN_RIGHT']:
                        try:
                            index_goal = info[object + f'_{k}'][f'{d}_vision'].index(3)
                        except ValueError:
                            None
                    # -- reward or punish for seeing the goal or not -- #
                    if index_goal!=-1:
                        
                        reward[object][k] += self.rewards_dist[object]['SEE_GOAL']/(1 + pow(info[object + f'_{k}']['dist'],1))
                    else:
                    
                        reward[object][k] += self.rewards_dist[object]['DONT_SEE_GOAL']
                    

                    #reward[k] += self.rewards_dist['DIST']/(1 + pow(info[f'agent_{k}']['dist'],1))

                    # --- reward/ punish for getting closer/farther from the goal --- #
                    if self.objects_dist[object + f'_{k}'][0]>self.objects_dist[object + f'_{k}'][1]:
                        reward[object][k]+=self.rewards_dist[object]['GET_CLOSER_CONSTANT']+self.rewards_dist[object]['GET_CLOSER']/(1 + pow(info[object + f'_{k}']['dist'],1))
                    else:
                        reward[object][k]+=self.rewards_dist[object]['GET_FARTHER_CONSTANT'] + self.rewards_dist[object]['GET_FARTHER'] * (1 + pow(info[object + f'_{k}']['dist'],1))


                    # --- reward for arriving at the goal --- #
                    if info[object + f'_{k}']['success']:
                        
                        reward[object][k] += self.rewards_dist[object]['GOAL']

                    #reward[k] = np.clip(reward[k],-1,1)

                    # -- normalize the rewards compared to the size of the maze -- #
                    reward[object][k] = reward[object][k]/(info['max_pos']+1)
                    #reward[k] = np.tanh(reward[k])

                    self.cum_rewards[object][k] += reward[object][k]
        
        #truncated = truncated or all(reward <-1*info['max_pos'] for reward in self.cum_rewards)
       
        return new_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for object in info['type_of_objects']:
            if 'goal' in info[object]:
                object_cum_rewards = []
                for k in range(info['n_'+object]):
                    self.objects_dist[object + f'_{k}'] = deque([info[object + f'_{k}']['dist']],maxlen=2)
                    self.objects_recent_loc[object + f'_{k}'] = deque([info[object + f'_{k}']['pos']],maxlen=3)
                    self.objects_past[object + f'_{k}'] = {info[object + f'_{k}']['pos'] : 1}
                    object_cum_rewards.append(0)
                self.cum_rewards[object] = object_cum_rewards

        return obs, info