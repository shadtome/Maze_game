import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import gymnasium as gym
import Maze_env.env
import device
import random
import os
import Maze_env.wrappers.rewards as rw
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML,display
import DQN.models as models
import json
import Maze_env

# -- baseline rewards -- #
baseline_rw = rw.reward_dist()


class maze_agents:
    def __init__(self,maze_model,vision,
                 action_type = 'full',
                 rewards_dist = baseline_rw,
                   **kwargs):
        """Initalize the base agent class, put the vision length to give the agents"""

        # -- vision of the agent -- #
        self.vision = vision

        # -- rewards distribution -- #
        self.rewards_dist = rewards_dist

        # -- shape of the CNN input -- #
        self.CNN_shape = (3,2*vision + 1, 2*vision + 1)

        # -- action type --#
        # -- full for all the cardinal directions and stop -- #
        if action_type == 'full':
            self.n_actions = 5 # stay and the cardinal directions
        elif action_type == 'cardinal':
            self.n_actions = 4 # Cardinal Directions
        self.action_type = action_type

        # -- initalize the maze model -- #
        self.maze_model = maze_model
        self.Q_fun = maze_model(self.CNN_shape,self.n_actions)
        self.Q_fun.to(device.DEVICE)
        
        # -- load the weights -- #
        if 'load' in kwargs.keys():
            self.Q_fun.load_state_dict(kwargs['load'])
        else:
            # -- randomly initalize the weights
            self.Q_fun.apply(self.weights_init)

        # -- save the last replay for the agents -- #
        self.__last_replay_agents_perspective__ = None

    @classmethod
    def load(cls,name):
        """Load the agents model"""
        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')
        fd = os.path.join(fd,f'{name}')

        # -- load model hyperparameters -- #
        with open(os.path.join(fd,'model_hyperparameters.json'),'r') as f:
            loaded_model_hp =json.load(f)
        name = loaded_model_hp['name']
        vision = loaded_model_hp['vision']
        action_type = loaded_model_hp['action_type']
        param_load = torch.load(os.path.join(fd, f'agent.pth'))

        # -- load reward distribution -- #
        with open(os.path.join(fd,'reward_distribution.json'),'r') as g:
            rewards = json.load(g)
        rewards_dist = rw.reward_dist(**rewards)
        result = cls(models.metadata[name],
                     vision,
                     action_type,
                     load = param_load,
                     rewards_dist = rewards_dist)
        return result
    
    def copy(self):
        copyAgent = maze_agents(self.maze_model,self.vision,self.action_type,
                                self.rewards_dist, load = self.Q_fun.state_dict())
        return copyAgent
        

    def transform_local_to_nn(self,local_state):
        """Used to transform information in enviroment numpy to Q-network pytorch,
                plus permute and unsqueeze dimensions"""
        
        result = torch.tensor(local_state,dtype=torch.float,device = device.DEVICE)
        
        result = result.permute(2,0,1)
        result = result.unsqueeze(0)
        result = result/255
        return result
    
    def transform_global_to_nn(self,global_state):
        """ Transform global information suitable for nn inputs """
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
        env = rw.maze_runner_rewards(env, rewards_dist = self.rewards_dist)
        return env

    def get_action(self,env,num_agents,state,epsilon=0.0):
        """ Get the actions from each agent from the state

            env: the environment the agent is in.

            num_agents: the number of agents in the environment

            state: the states the agents are in

            epsilon: the probability of taking a random action vs 1-epsilon to take 
                    a Q-network action"""
        
        actions = []
        for a in range(num_agents):
            actions.append(self.get_single_agent_action(env,state,a,epsilon))       
        return actions
    
    def get_single_agent_action(self,env,state,a,epsilon):
        if np.random.random()<epsilon:
            action = int(env.action_space.sample())
        else:
            
            local_state_tensor = self.transform_local_to_nn(state[f'local_{a}'])
            global_state_tensor = self.transform_global_to_nn(state[f'global_{a}'])
            q_values = self.Q_fun(local_state_tensor,global_state_tensor)
            action=int(q_values.argmax().item())
        return action
    
    def compute_action_probs(self,local_state,global_state):
        """ Compute the probabilities of each action from the state using Q-Net"""
        local_state_tensor = self.transform_local_to_nn(local_state)
        global_state_tensor = self.transform_global_to_nn(global_state)
        q_values = self.Q_fun(local_state_tensor,global_state_tensor)
        
        #q_values = self.Q_fun(state_tensor)
        action_probs = torch.softmax(q_values,dim=1)
        return action_probs
        
        
    def get_replay(self,env,num_agents,state,epsilon):
        """ Get replay information from an action
            
            env: gymnasium environment for the maze runner
            
            num_agents: total number of agents in the environment
            
            state: current state
            
            epsilon: probability of giving a random action"""

        actions = self.get_action(env,num_agents,state,epsilon)

        next_state, reward, terminated, truncated, info = env.step(actions)

        return state, actions, next_state, reward, terminated
    
    def run_agent(self,maze, len_game = 50, n_episodes = 1, num_agents = 1,epsilon = 0, sample_prob = False,
                  agents_pos=None, targets_pos = None, start_dist = None,
                  output_frame_rewards = False):
        """Run the agent in the enviroment that is human readable using pygame.

            maze: a maze from the maze_dataset, needs the connection_list,

            n_episodes: number of episodes for the agent to go through

            len_game: max length of steps in the game

            num_agents: the number of agents in the enviroment with the same Q-net

            epsilon: the probability of using a random action

            sample_prob: outputs the probabilities of actions from the states
            
            agents_pos: position of the agents in a list, where index corresponds
                        to which agent
            targets_pos: position of the agent's target in a list, where index corresponds
                        to which agents' target."""
        
        # -- save agents perspective for viewing later -- #
        agents_per = {}
        self.Q_fun.eval()
        with torch.no_grad():
            # -- make environment -- #
            env = gym.make('Maze_env/MazeRunner-v0',len_game = len_game,num_agents=num_agents,vision_len=self.vision,maze=maze,
                           render_mode='human',obs_type = 'spatial',
                           action_type = self.action_type, 
                           agents_pos = agents_pos, targets_pos = targets_pos,
                           start_dist = start_dist)

            env = self.add_wrappers(env)

            # -- go through the number of episodes and enact the environment -- #
            for i in range(n_episodes):
                obs, info = env.reset()
                for a in range(num_agents):
                    agents_per[f'agent_{a}'] = [obs[f'local_{a}']]

                done = False

                cum_reward = 0
                frame = 0
                # Play
                while not done:
                    # -- get actions -- #
                    action = self.get_action(env,num_agents,obs,epsilon)
                    # -- get next observations -- #
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    for a in range(num_agents):
                        agents_per[f'agent_{a}'].append(next_obs[f'local_{a}'])
                    cum_reward += reward

                    if sample_prob == True:
                        pic = self.compute_action_probs(next_obs['local_0'])
                        print(pic.numpy())
                    # -- output frame rewards -- # 
                    if output_frame_rewards:
                        frame+=1
                        print(f'Frame: {frame} reward: {reward}')
                    # -- done -- #
                    done = terminated or truncated
                    obs = next_obs
                    self.__last_replay_agents_perspective__ = agents_per
                print(f'cumulative reward: {cum_reward}')
            env.close()
            self.__last_replay_agents_perspective__ = agents_per

    def test_agent(self,maze_dataset, n_episodes,len_game = 50,
                    num_agents = 1,
                  agents_pos=None, targets_pos = None,
                  start_dist = None):
        """This evaluates how good the agent is at random maze levels
            Returns the ratio completed episodes/total number of episodes""" 
        total_completed = 0
        

        for i in range(len(maze_dataset)):
            id_x = random.choice(range(len(maze_dataset)))
            maze = maze_dataset[id_x]
            self.Q_fun.eval()
            with torch.no_grad():
                # -- make environment -- #
                env = gym.make('Maze_env/MazeRunner-v0',len_game = len_game,
                            num_agents=num_agents,vision_len=self.vision,maze=maze,
                            render_mode='rgb_array',obs_type = 'spatial',
                            action_type = self.action_type, 
                            agents_pos = agents_pos, targets_pos = targets_pos,
                            start_dist = start_dist)

                env = self.add_wrappers(env)

                # -- go through the number of episodes and enact the environment -- #
                for i in range(n_episodes):
                    obs, info = env.reset()

                    done = False

                    # Play
                    while not done:
                        # -- get actions -- #
                        action = self.get_action(env,num_agents,obs,epsilon=0)
                        # -- get next observations -- #
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        
                        
                        # -- done -- #
                        done = terminated or truncated
                        obs = next_obs
                        if done and info['agent_0']['pos'] == info['agent_0']['target']:
                            total_completed+=1
                env.close()
        return total_completed/n_episodes
               

    def animate_last_replay(self,agent_id,name, save = False):
        """Takes the last run_agent and saved perspectives of the agents and 
            returns a animation of it
            agent_id: gives the id of the agent for the perspective we want
            name: name of the replay to save in the media folder"""
        
        seq_anim = self.__last_replay_agents_perspective__[f'agent_{agent_id}']
        html, ani = create_animation(seq_anim)
        display(html)
        if save:
            fd = os.getcwd()
            fd = os.path.join(fd, 'media')
            if os.path.exists(fd) == False:
                os.mkdir(fd)
            fd = os.path.join(fd,f'{name}.gif')
            ani.save(fd, writer='pillow')
                  

    def save(self,name):
        """Save the agents model"""
        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')

        if os.path.exists(fd)==False:
            os.mkdir(fd)

        fd = os.path.join(fd,f'{name}')
        if os.path.exists(fd)==False:
            os.mkdir(fd)
        # -- save the agent model -- #
        torch.save(self.Q_fun.state_dict(),os.path.join(fd,f'agent.pth'))

        model_param = {
            'name': self.Q_fun.name,
            'vision': self.vision,
            'action_type': self.action_type
        }
        # -- save the type of model with other agent specific parameters -- #
        with open(os.path.join(fd,'model_hyperparameters.json'),'w') as f:
            json.dump(model_param, f, indent=4)

        reward_structure = {
            'GOAL' : self.rewards_dist['GOAL'],
            'SEE_GOAL': self.rewards_dist['SEE_GOAL'],
            'DONT_SEE_GOAL': self.rewards_dist['DONT_SEE_GOAL'],
            'NEW_PLACE' : self.rewards_dist['NEW_PLACE'],
            'OLD_PLACE' : self.rewards_dist['OLD_PLACE'],
            'GET_CLOSER': self.rewards_dist['GET_CLOSER'],
            'GET_FARTHER': self.rewards_dist['GET_FARTHER'],
            'DIST': self.rewards_dist['DIST'],
            'WALL' : Maze_env.env.mazes.WALL
        }
        # -- save reward distribution -- #
        with open(os.path.join(fd,'reward_distribution.json'),'w') as f:
            json.dump(reward_structure,f,indent=4)

        

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
    return HTML(anim.to_jshtml()), anim