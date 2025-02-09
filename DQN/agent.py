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
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        h = state_shape[1]
        w = h
       # nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            #nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        self.Q_function = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Flatten(),
            nn.Linear(32*h*w,32),
            nn.ReLU(),
            nn.Linear(32,n_actions)
        )

    def forward(self,x):
        #x = self.normalize(x)
        x = self.Q_function(x)
        return x
    
    
    def copy(self):
        copy = CNN_Q_fun(self.state_shape,self.n_actions)
        copy.load_state_dict(self.state_dict())
        return copy

class CNN_Maze_Agents:
    def __init__(self,vision):
        """Initalize the base agent class, put the vision length to give the agents"""

        ################
        # Enviroment info
        ## vision length of the agents
        self.vision = vision

        

        ####################
        # state shape (3, 2*vision + 1, 2*vision + 1) 
        # Local spatial images of the agents location 
        self.state_shape = (3,2*vision + 1, 2*vision + 1)
        self.n_actions = 5 # stay and the cardinal directions

        #Define Q_function neural network
        self.Q_fun = CNN_Q_fun(self.state_shape,self.n_actions)
        self.Q_fun.to(device.DEVICE)
        #self.Q_fun.apply(self.weights_init)

        for layer in self.Q_fun.Q_function:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)



        ##################
        # Animation and replay purposes
        self.__last_replay_agents_perspective__ = None

    def transform_to_nn(self,state):
        result = torch.tensor(state,dtype=torch.float,device = device.DEVICE)
        
        result = result.permute(2,0,1)
        result = result.unsqueeze(0)
        result = (result - 127.5)/127.5
        return result
    
    def transform_to_env(self,state):
        result = state.squeeze(1).numpy()
        result = result.permute(1,2,0)
        result = result.numpy()
        return result

    def add_wrappers(self, env):
        env = rw.maze_runner_rewards(env)
        return env

    def get_action(self,env,num_agents,state,epsilon=0):
        actions = []
        for a in range(num_agents):
            if np.random.random()<epsilon:
                actions.append(int(env.action_space.sample()))
            else:
                
                state_tensor = self.transform_to_nn(state[f'local_{a}'])
                q_values = self.Q_fun(state_tensor)
                actions.append(int(q_values.argmax().item()))
                
        return actions
    
    def compute_action_probs(self,state):
        
        state_tensor = self.transform_to_nn(state)
        q_values = self.Q_fun(state_tensor)
        
        #q_values = self.Q_fun(state_tensor)
        action_probs = torch.softmax(q_values,dim=1)
        return action_probs
        
        
    def get_replay(self,env,num_agents,state,epsilon):

        actions = self.get_action(env,num_agents,state,epsilon)

        next_state, reward, terminated, truncated, info = env.step(actions)

        return state, actions, next_state, reward, terminated
    
    def run_agent(self,maze, len_game = 1000, num_agents = 1,epsilon = 0, sample_prob = False):
        agents_per = {}
        
        with torch.no_grad():
            # make enviroment for testing
            env = gym.make('Maze_env/MazeRunner-v0',len_game = len_game,num_agents=num_agents,vision_len=self.vision,maze=maze,
                           render_mode='human',obs_type = 'spatial')

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
        seq_anim = self.__last_replay_agents_perspective__[f'agent_{agent_id}']
        html = create_animation(seq_anim)
        display(html)

    def save(self,filepath,name):
        torch.save(self.Q_fun.state_dict(),os.path.join(filepath,f'{name}.pth'))

    def load(self,filepath,name):
        self.Q_fun.load_state_dict(torch.load(os.path.join(filepath, f'{name}.pth')))

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data,0.0,0.02)
        elif classname.find('BatchNorm') !=-1:
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)



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