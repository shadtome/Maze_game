import numpy as np
import torch
import torch.nn as nn

class Agent_Q_fun(nn.Module):
    def __init__(self,state_shape,n_actions):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q_function = nn.Sequential(
            nn.Linear(state_shape,16),
            nn.ReLU(),
            nn.Linear(16,n_actions),
            nn.ReLU()
        )

    def forward(self,x):
        return self.Q_function(x)

class maze_agent:
    def __init__(self, start_loc, end_loc, agent_id):
        self.agent_id = agent_id
        
        self.loc = start_loc
        self.goal = end_loc

        # Define Q neural network
        self.Q_fun = Agent_Q_fun(7,5)


    def get_action(self,env,state,epsilon=0):
        if np.random.random()<epsilon:
            return int(env.action_space.sample())
        else:
            state_tensor = torch.tensor(np.array([state]),dtype = torch.float32)
            return int(self.Q_fun(state_tensor).argmax().item())
        
    def get_replay(self,env,state,epsilon):
        action = self.get_action(env,state,epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)
        return state,action, next_state, reward, terminated
    
