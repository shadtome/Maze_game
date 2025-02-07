import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import device
import os

class Agent_Q_fun(nn.Module):
    def __init__(self,state_shape,n_actions):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q_function = nn.Sequential(
            nn.Linear(state_shape,16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(16,n_actions),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self,x):
        return self.Q_function(x)

class Maze_Agents:
    def __init__(self,vision):
        """Initalize the base agent class, put the vision length to give the agents"""

        # get the state shapes and 
        self.state_shape = (3 + 12*vision) # (pos 2) + (dis 1) + (objects see 3)*(dir 4)*vision
        self.n_actions = 5 # stay and the cardinal directions
        #Define Q_function neural network
        self.Q_fun = Agent_Q_fun(self.state_shape,self.n_actions)
        self.Q_fun.to(device.DEVICE)

    def __combineObsInfo__(self,obs,info):
        directions  = {'UP','DOWN','LEFT','RIGHT'}
        vision = []
        for dir in directions:
            vision.append(np.eye(4)[info[f'agent_{self.id}'][f'{dir}_vision']])
        vision_stack = np.hstack(vision)
        x,y = obs[f'agent_{self.id}']
        dis = info[f'agent_{self.id}']['man_dist']
        state = np.hstack([vision_stack.flatten(),[x,y],[dis]])
        return state

    def add_wrappers(self, env):
        return env

    def get_action(self,env,state,epsilon=0):
        
        if np.random.random()<epsilon:
            return int(env.action_space.sample())
        else:
            
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32,device=device.DEVICE)
            return int(self.Q_fun(state_tensor).argmax().item())
        
    def get_replay(self,env,state,epsilon):

        action = self.get_action(env,state,epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        return state, action, next_state, reward, terminated
    
    def run_agent(self,num_agents,maze):
        with torch.no_grad():
            # make enviroment for testing
            env = gym.make('Maze_env/MazeRunner-v0',num_agents=num_agents,vision_len=3,maze=maze,
                           render_mode='human',continuous=False)

            env = self.add_wrappers(env)

            obs, info = env.reset()

            done = False

            cum_reward = 0
            # Play
            while not done:
                
                action = self.get_action(env,obs,epsilon=0)
            
                next_obs, reward, terminated, truncated, info = env.step(action)
                cum_reward += reward

                done = terminated or truncated
                obs = next_obs
            env.close()
            print(f'cumulative reward: {cum_reward}')

    def save(self,filepath):
        torch.save(self.Q_fun.state_dict(),os.path.join(filepath,'final_agent.pth'))

    def load(self,filepath):
        self.Q_fun.load_state_dict(torch.load(os.path.join(filepath, 'final_agent.pth')))