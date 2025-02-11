import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
import device
import os
import Maze_env.wrappers.rewards as rw
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML,display
import Maze_env

class CNN_Q_fun(nn.Module):
    def __init__(self,state_shape,n_actions):
        """THe Convolutional Neural network
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__()
        self.state_shape = state_shape # Shape of the images for the local information
        self.n_actions = n_actions # number of actions
        h = state_shape[1]
        w = state_shape[2]


       
        self.CNN_function= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*int((h-3)/2)*int((w-3)/2),32),
            nn.ReLU(),
        )
        # Takes inputs of the form (pos,g_pos,done,dist)
        self.global_function = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,12),
            nn.ReLU(),
        )

        self.final_function =nn.Sequential(
            nn.Linear(32 + 12,12),
            nn.ReLU(),
            nn.Linear(12,self.n_actions)
        ) 

    def forward(self,x,y):
        
        x = self.CNN_function(x)
        y = self.global_function(y)
        
        combined = torch.cat((x,y),dim=-1)
        result = self.final_function(combined)
        return result
    
    
    def copy(self):
        q_copy = CNN_Q_fun(self.state_shape,self.n_actions)
        q_copy.load_state_dict(self.state_dict())
        return q_copy

class CNN_Maze_Agents:
    def __init__(self,vision,action_type = 'full'):
        """Initalize the base agent class, put the vision length to give the agents"""

        ################
        # Enviroment info
        ## vision length of the agents
        self.vision = vision

        ####################
        # state shape (3, 2*vision + 1, 2*vision + 1) 
        # Local spatial images of the agents location 
        self.CNN_shape = (3,2*vision + 1, 2*vision + 1)
        if action_type == 'full':
            self.n_actions = 5 # stay and the cardinal directions
        elif action_type == 'cardinal':
            self.n_actions = 4 # Cardinal Directions
        self.action_type = action_type

        #Define Q_function neural network
        self.Q_fun = CNN_Q_fun(self.CNN_shape,self.n_actions)
        self.Q_fun.to(device.DEVICE)

        ######################
        # Used to randomly initalizze the weights
        self.Q_fun.apply(self.weights_init)


        #for layer in self.Q_fun.Q_function:
         #   if isinstance(layer, nn.Linear):
          #      nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
           #     if layer.bias is not None:
            #        nn.init.zeros_(layer.bias)



        ##################
        # Animation and replay purposes
        self.__last_replay_agents_perspective__ = None

    def transform_local_to_nn(self,local_state):
        """Used to transform information in enviroment numpy to Q-network pytorch,
                plus permute and unsqueeze dimensions"""
        
        result = torch.tensor(local_state,dtype=torch.float,device = device.DEVICE)
        
        result = result.permute(2,0,1)
        result = result.unsqueeze(0)
        result = result/255
        return result
    
    def transform_global_to_nn(self,global_state):
        result = torch.tensor(global_state,dtype=torch.float32,device=device.DEVICE)
        result[0:2] = result[0:2]/(self.CNN_shape[1]*self.CNN_shape[2] -1)
        result[3] = result[3]/(self.CNN_shape[1] + self.CNN_shape[2])
        result = result.unsqueeze(0)
        return result
    
    def transform_local_to_env(self,local_state):
        """ Transform Q-network pytorch information to the enviroment numpy"""
        result = local_state.squeeze(1).numpy()
        result = result.permute(1,2,0)
        result = result.numpy()
        return result
    
    def transform_global_to_env(self,global_state):
        return global_state.numpy()

    def add_wrappers(self, env):
        """Add wrappers into the enviroment"""
        env = rw.maze_runner_rewards(env)
        return env

    def get_action(self,env,num_agents,state,epsilon=0):
        """ Get the actions from each agent from the state
            env: the environment the agent is in.
            num_agents: the number of agents in the environment
            state: the states the agents are in
            epsilon: the probability of taking a random action vs 1-epsilon to take 
                    a Q-network action"""
        actions = []
        for a in range(num_agents):
            if np.random.random()<epsilon:
                actions.append(int(env.action_space.sample()))
            else:
                
                local_state_tensor = self.transform_local_to_nn(state[f'local_{a}'])
                global_state_tensor = self.transform_global_to_nn(state[f'global_{a}'])
                q_values = self.Q_fun(local_state_tensor,global_state_tensor)
                actions.append(int(q_values.argmax().item()))
                
        return actions
    
    def compute_action_probs(self,local_state,global_state):
        """ Compute the probabilities of each action from the state using Q-Net"""
        local_state_tensor = self.transform_local_to_nn(local_state)
        global_state_tensor = self.transform_global_to_nn(global_state)
        q_values = self.Q_fun(local_state_tensor,global_state_tensor)
        
        #q_values = self.Q_fun(state_tensor)
        action_probs = torch.softmax(q_values,dim=1)
        return action_probs
        
        
    def get_replay(self,env,num_agents,state,epsilon):
        """ Get replay information from an action"""

        actions = self.get_action(env,num_agents,state,epsilon)

        next_state, reward, terminated, truncated, info = env.step(actions)

        return state, actions, next_state, reward, terminated
    
    def run_agent(self,maze, len_game = 1000, num_agents = 1,epsilon = 0, sample_prob = False):
        """Run the agent in the enviroment that is human readable using pygame.
            maze: a maze from the maze_dataset, needs the connection_list,
            len_game: max length of steps in the game
            num_agents: the number of agents in the enviroment with the same Q-net
            epsilon: the probability of using a random action
            sample_prob: outputs the probabilities of actions from the states"""
        
        #####################
        # Agents perspective, saved for observation after playing in the environment
        agents_per = {}
        
        with torch.no_grad():
            # make enviroment for testing
            env = gym.make('Maze_env/MazeRunner-v0',len_game = len_game,num_agents=num_agents,vision_len=self.vision,maze=maze,
                           render_mode='human',obs_type = 'spatial',
                           action_type = self.action_type)

            env = self.add_wrappers(env)

            obs, info = env.reset()
            for a in range(num_agents):
                agents_per[f'agent_{a}'] = [obs[f'local_{a}']]

            done = False

            cum_reward = 0
            # Play
            while not done:
                # Get each of the agents action
                action = self.get_action(env,num_agents,obs,epsilon)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                for a in range(num_agents):
                    agents_per[f'agent_{a}'].append(next_obs[f'local_{a}'])
                cum_reward += reward

                if sample_prob == True:
                    pic = self.compute_action_probs(next_obs['local_0'])
                    print(pic.numpy())

                done = terminated or truncated
                obs = next_obs
                self.__last_replay_agents_perspective__ = agents_per

            env.close()
            self.__last_replay_agents_perspective__ = agents_per
            print(f'cumulative reward: {cum_reward}')

    def animate_last_replay(self,agent_id):
        """Takes the last run_agent and saved perspectives of the agents and 
            returns a animation of it
            agent_id: gives the id of the agent for the perspective we want"""
        seq_anim = self.__last_replay_agents_perspective__[f'agent_{agent_id}']
        html = create_animation(seq_anim)
        display(html)

    def save(self,name):
        """Save the agents model"""
        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')

        if os.path.exists(fd)==False:
            os.mkdir(fd)

        fd = os.path.join(fd,f'{name}')
        if os.path.exists(fd)==False:
            os.mkdir(fd)
        
        torch.save(self.Q_fun.state_dict(),os.path.join(fd,f'agent.pth'))

    def load(self,name):
        """Load the agents model"""
        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')
        fd = os.path.join(fd,f'{name}')
        self.Q_fun.load_state_dict(torch.load(os.path.join(fd, f'agent.pth')))

    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming He for conv layers
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Bias = 0 (standard practice)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming He for fully connected layers
            if m.bias is not None:
                nn.init.zeros_(m.bias)



def create_animation(image_sequence, interval=200):
    """
    Creates an animation from a sequence of images in a Jupyter Notebook.

    Args:
        image_sequence (list of numpy arrays): List of images (HxW or HxWxC).
        interval (int): Delay between frames in milliseconds.

    Returns:
        HTML: Animation rendered as an HTML object.
    """
    fig, ax = plt.subplots()

    # Show the first frame
    img_display = ax.imshow(image_sequence[0], cmap='gray', animated=True)

    def update(frame):
        """Updates the image for each frame."""
        img_display.set_array(image_sequence[frame])
        return img_display,

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(image_sequence), interval=interval, blit=True)

    # Display animation as HTML in Jupyter Notebook
    return HTML(anim.to_jshtml())