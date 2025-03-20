from gymnasium import Wrapper
from collections import deque
import numpy as np






# --- Reward Hyperparameters --- #
GOAL = 40.0
SEE_GOAL = 0.99
DONT_SEE_GOAL = -0.5
NEW_PLACE = 0.4
OLD_PLACE = -0.6
GET_CLOSER = 0.5
GET_FARTHER = -0.6
DIST = 0.0

class reward_dist:
    def __init__(self, **kwargs):
        """Set up the reward distribution for the agent.
        The following are the possible reward parameters
        GOAL:
        SEE_GOAL:
        DONT_SEE_GOAL:
        NEW_PLACE:
        OLD_PLACE:
        GET_CLOSER:
        GET_CLOSER_CONSTANT:
        GET_FARTHER:
        GET_FARTHER_CONSTANT:
        DIST:
        HIT_OTHER:
        TOO_CLOSE:
        TOO_CLOSE_CONSTANT:
        """
        # -- Reward parameters -- #
        self.rewards = {
            'GOAL': 1.0,
            'SEE_GOAL': 0.0,
            'DONT_SEE_GOAL': -0.0,
            'NEW_PLACE': 0.0,
            'OLD_PLACE': -0.0,
            'GET_CLOSER': 0.0,
            'GET_CLOSER_CONSTANT': 0.0,
            'GET_FARTHER': -0.0,
            'GET_FARTHER_CONSTANT': -0.0,
            'DIST': 0.0,
            'HIT_OTHER': -1.0,
            'TOO_CLOSE': -0.0,
            'TOO_CLOSE_CONSTANT': -0.0,
        }
        self.timer = 0
        self.decay_rate = 0
        self.change_rewards(**kwargs)

    def change_rewards(self, **kwargs):
        """Takes any of the key words for the 
        goals and changes them."""
        for k in kwargs:
            if k in self.rewards:
                self.rewards[k] = kwargs[k]

    def __getitem__(self,item):
        return self.rewards[item]
    
    def __setitem__(self,key,value):
        self.rewards[key] = value

    def step(self):
        None
        

    

class maze_runner_rewards(Wrapper):
    def __init__(self,env, rewards_dist):

        super().__init__(env)

        # -- rewards distribution -- #
        self.rewards_dist = rewards_dist
        
        self.agents_dist = {}
        self.agents_recent_loc = {}
        self.agents_past = {}

        self.cum_rewards = []


    def step(self,action):
        new_obs, reward, terminated, truncated, info = super().step(action)

    
        for k in range(info['n_agents']):
            self.agents_dist[f'agent_{k}'].append(info[f'agent_{k}']['dist'])

            # --- positions of the agent and the goal --- #
            pos = info[f'agent_{k}']['pos']
            t_pos = info[f'agent_{k}']['target']
            agent_done = info[f'agent_{k}']['done']
            self.agents_recent_loc[f'agent_{k}'].append(pos)

            # --- punish for revisiting spots --- #
            if pos not in self.agents_past[f'agent_{k}']:
                self.agents_past[f'agent_{k}'][pos] = 1
                
                reward[k] +=self.rewards_dist['NEW_PLACE']/(1 + pow(info[f'agent_{k}']['dist'],1))
            else:
                # --- punish corresponding to how many times it visited --- #
                self.agents_past[f'agent_{k}'][pos]+=1
                reward[k]+=self.rewards_dist['OLD_PLACE']* float(self.agents_past[f'agent_{k}'][pos])

            # --- check neighborhoods for goals and other agents --- #
            index_goal = -1
            index_agent = +1000
            for d in ['CENTER','UP', 'DOWN','LEFT','RIGHT','UP_LEFT','UP_RIGHT','DOWN_LEFT','DOWN_RIGHT']:
                try:
                    index_goal = info[f'agent_{k}'][f'{d}_vision'].index(3)
                except ValueError:
                    None
                try:
                    index_agent = min(index_agent,info[f'agent_{k}'][f'{d}_vision'].index(1))
                except ValueError:
                    None
            # -- reward or punish for seeing the goal or not -- #
            if index_goal!=-1:
                
                reward[k] += self.rewards_dist['SEE_GOAL']/(1 + pow(info[f'agent_{k}']['dist'],1))
            else:
               
                reward[k] += self.rewards_dist['DONT_SEE_GOAL']
            
            # -- discourge getting closer to other agents -- #
            if index_agent!=1000:
                reward[k] += self.rewards_dist['TOO_CLOSE_CONSTANT'] + self.rewards_dist['TOO_CLOSE']/(1 + index_agent)
            

            #reward[k] += self.rewards_dist['DIST']/(1 + pow(info[f'agent_{k}']['dist'],1))

            # --- reward/ punish for getting closer/farther from the goal --- #
            if self.agents_dist[f'agent_{k}'][0]>self.agents_dist[f'agent_{k}'][1]:
                reward[k]+=self.rewards_dist['GET_CLOSER_CONSTANT']+self.rewards_dist['GET_CLOSER']/(1 + pow(info[f'agent_{k}']['dist'],1))
            else:
                reward[k]+=self.rewards_dist['GET_FARTHER_CONSTANT'] + self.rewards_dist['GET_FARTHER'] * (1 + pow(info[f'agent_{k}']['dist'],1))


            # --- reward for arriving at the goal --- #
            if pos==t_pos and agent_done and info[f'agent_{k}']['dead']==False:
                reward[k] += self.rewards_dist['GOAL']
            # --- punish for hitting other agents --- #
            elif info[f'agent_{k}']['dead']:
                reward[k] += self.rewards_dist['HIT_OTHER']

            #reward[k] = np.clip(reward[k],-1,1)

            # -- normalize the rewards compared to the size of the maze -- #
            reward[k] = reward[k]/(info['max_pos']+1)
            #reward[k] = np.tanh(reward[k])

            self.cum_rewards[k] += reward[k]
        
        #truncated = truncated or all(reward <-1*info['max_pos'] for reward in self.cum_rewards)
       
        return new_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        for a in range(info['n_agents']):
            self.agents_dist[f'agent_{a}'] = deque([info[f'agent_{a}']['dist']],maxlen=2)
            self.agents_recent_loc[f'agent_{a}'] = deque([info[f'agent_{a}']['pos']],maxlen=3)
            self.agents_past[f'agent_{a}'] = {info[f'agent_{a}']['pos'] : 1}
            self.cum_rewards.append(0)
        

        return obs, info