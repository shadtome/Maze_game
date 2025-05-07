import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import gymnasium as gym
import Maze_env.env
import device
import random
import os
from Maze_env.reward_functions.basic import BasicRewardFun
import Maze_env.wrappers.reward_wrappers.runner_rewards as rw
from Maze_env.game_info.basic_info import basicGame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML,display
import DQN.models.base as base
import json
import Maze_env
import imageio
import pygame

# -- baseline rewards -- #
baseline_rw = BasicRewardFun()


class BaseAgent:
    def __init__(self,model,vision,
                 action_type = 'full',
                 dist_paradigm = 'radius',
                 game_info = basicGame(),
                   **kwargs):
        """Initalize the base agent class, put the vision length to give the agents"""

        # -- game info -- #
        self.game_info = game_info


        # -- vision of the agent -- #
        self.vision = vision

        # -- type of distance to goal -- #
        self.dist_paradigm = dist_paradigm


        # -- shape of the CNN input -- #
        self.CNN_shape = {}
        for k,v in vision.items():
            self.CNN_shape[k] = (3,2*v + 1, 2*v + 1)
        

        # -- action type --#
        # -- full for all the cardinal directions and stop -- #
        if action_type == 'full':
            self.n_actions = 5 # stay and the cardinal directions
        elif action_type == 'cardinal':
            self.n_actions = 4 # Cardinal Directions
        self.action_type = action_type

        # -- initalize the maze model -- #
        self.maze_model = model
        self.Q_fun = {}
        for k in self.game_info.type_of_objects:
            self.Q_fun[k] = self.__init_model__(model[k],self.CNN_shape[k],self.n_actions, **kwargs)
            self.Q_fun[k].to(device.DEVICE)
        
        # -- load the weights -- #
        if 'load' in kwargs.keys():
            for k,v in self.Q_fun.items():
                v.load_state_dict(kwargs['load'][k])
        else:
            # -- randomly initalize the weights
            for v in self.Q_fun.values():
                v.apply(self.weights_init)

        # -- save the last replay for the agents -- #
        self.__last_replay_agents_perspective__ = None

        # -- record the pygame output -- #
        self.__pygame_output__ = None

    @classmethod
    def load(cls,name, game_info,rewards_cls = BasicRewardFun,
             default_rewards_dist = None):
        """Load the agent's model dynamically, supporting inheritance."""

        models = {}
        visions = {}
        param_load = {}
        rewards_dist = {}
        action_type = None
        dist_paradigm = None
        

        fd = os.getcwd()
        fd = os.path.join(fd, 'trained_agents', name)


        # -- Load model hyperparameters -- #
        
        for type_object in game_info.type_of_objects:
            object_fd = os.path.join(fd, type_object)
            with open(os.path.join(object_fd, 'model_hyperparameters.json'), 'r') as f:
                loaded_model_hp = json.load(f)

            models[type_object] = base.metadata[loaded_model_hp['model_name']]
            visions[type_object] = loaded_model_hp['vision']
            action_type = loaded_model_hp['action_type']
            dist_paradigm = loaded_model_hp['dist_paradigm']
            param_load[type_object] = torch.load(os.path.join(object_fd, 'agent.pth'))

            # -- Load reward distribution -- #
            if default_rewards_dist!=None:
                rewards_dist[type_object] = default_rewards_dist[type_object]
            else:
                with open(os.path.join(object_fd, 'reward_distribution.json'), 'r') as g:
                    rewards = json.load(g)
                rewards_dist[type_object] = rewards_cls(**rewards)
            game_info.rewards_dist = rewards_dist
            # -- handle additional parameters unkown to this class -- #
            #extra_params = {k: v for k, v in loaded_model_hp.items() if k not in base_params}
        
        return cls(model=models,vision=visions,action_type=action_type,
                   dist_paradigm = dist_paradigm,
                   game_info = game_info,
                   load = param_load)
    
    def load_object(self,obj_type,name,file_obj_name):
        """ This is used to load in the parameters for a certain object"""
        fd = os.getcwd()
        fd = os.path.join(fd, 'trained_agents', name)
        object_fd = os.path.join(fd, file_obj_name)
        param_load = torch.load(os.path.join(object_fd, 'agent.pth'))

        self.Q_fun[obj_type].load_state_dict(param_load)

    def copy(self):
        load = {}
        for k,v in self.Q_fun.items():
            load[k] = v.state_dict()
        copyAgent = self.__class__(model = self.maze_model,
                                   vision = self.vision,
                                   action_type = self.action_type,
                                    dist_paradigm = self.dist_paradigm, 
                                    game_info = self.game_info,
                                    load = load)
        return copyAgent
    
    def set_game(self,game_info):
        self.game_info = game_info
        
    def __init_model__(self, maze_model,CNN_shape,n_actions, **kwargs):
        return maze_model(CNN_shape,n_actions)
    
    def max_dist(self,maze_dataset):
        n_rows = maze_dataset.shape[0]
        n_cols = maze_dataset.shape[1]
        if self.dist_paradigm == "radius":
            return n_rows + n_cols -2
        if self.dist_paradigm == 'path':
            return n_rows*n_cols - 1

    def transform_local_to_nn(self,local_state):
        """Used to transform information in enviroment numpy to Q-network pytorch,
                plus permute and unsqueeze dimensions"""
        
        result = torch.tensor(local_state,dtype=torch.float,device = device.DEVICE)
        
        result = result.permute(2,0,1)
        result = result.unsqueeze(0)
        result = result/255
        return result
    
    def transform_global_to_nn(self,global_state,type_object):
        """ Transform global information suitable for nn inputs """
        result = torch.tensor(global_state,dtype=torch.float32,device=device.DEVICE)
        result[0:2] = result[0:2]/(self.CNN_shape[type_object][1]*self.CNN_shape[type_object][2] -1)
        result[3] = result[3]/(self.CNN_shape[type_object][1] + self.CNN_shape[type_object][2])
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
        """Add wrappers into the enviroment\n
        Mean to be changed for inheritence"""
        env = self.game_info.add_wrapper(env)
        return env

    def get_action(self,env,num_objects,state,info,epsilon=0.0,training_on=None):
        """ Get the actions from each agent from the state

            env: the environment the agent is in.

            num_agents: the number of agents in the environment

            state: the states the agents are in

            epsilon: the probability of taking a random action vs 1-epsilon to take 
                    a Q-network action"""

        actions = {}
        for obj_type in self.game_info.type_of_objects:
            action = []
            for a in range(num_objects[obj_type]):
                if training_on == None or training_on[obj_type]==True:
                    action.append(self.get_single_agent_action(env,state,a,info,epsilon,obj_type)) 
                else:
                    action.append(self.get_single_agent_action(env,state,a,info,0,obj_type)) 
            actions[obj_type] = action     
        return actions
    
    def get_single_agent_action(self,env,state,a,info,epsilon,type_object):
        if np.random.random()<epsilon:
            action = int(env.action_space.sample())
        else:
            
            local_state_tensor = self.transform_local_to_nn(state[type_object][f'local_{a}'])
            global_state_tensor = self.transform_global_to_nn(state[type_object][f'global_{a}'],type_object)
            q_values = self.Q_fun[type_object](local_state_tensor,global_state_tensor)
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
    
    def __make_env__(self,len_game,num_objects,maze,init_pos,start_dist,render_mode = 'rgb_array'):
        env = gym.make(self.game_info.maze_environment,
                           len_game = len_game,
                           num_objects=num_objects,
                           vision_len=self.vision,maze=maze,
                           render_mode=render_mode,obs_type = 'spatial',
                           action_type = self.action_type, 
                           init_pos = init_pos,
                           start_dist = start_dist,
                           collision_rules = self.game_info.collision_rules,
                           dist_paradigm = self.dist_paradigm,
                           type_of_objects = self.game_info.type_of_objects,
                           objectives = self.game_info.objectives,
                           colormap = self.game_info.colormap)

        env = self.add_wrappers(env)
        return env
        
        
    def get_replay(self,env,num_objects,state,epsilon,type_object):
        """ Get replay information from an action
            
            env: gymnasium environment for the maze runner
            
            num_agents: total number of agents in the environment
            
            state: current state
            
            epsilon: probability of giving a random action"""

        actions = self.get_action(env,num_objects,state,epsilon,type_object)

        next_state, reward, terminated, truncated, info = env.step(actions)

        return state, actions, next_state, reward, terminated
    
    def run_agent(self,maze, len_game = 50, n_episodes = 1, num_objects = {'agents':1},
                  epsilon = 0,
                  init_pos={}, start_dist = None,
                  output_frame_rewards = False,):
        """Run the agent in the enviroment that is human readable using pygame.

            maze: a maze from the maze_dataset, needs the connection_list,\n

            n_episodes: number of episodes for the agent to go through\n

            len_game: max length of steps in the game\n

            num_agents: the number of agents in the enviroment with the same Q-net\n

            epsilon: the probability of using a random action\n

            sample_prob: outputs the probabilities of actions from the states\n
            
            agents_pos: position of the agents in a list, where index corresponds
                        to which agent\n
            targets_pos: position of the agent's target in a list, where index corresponds
                        to which agents' target."""
        
        # -- save agents perspective for viewing later -- #
        objects_per = {}
        for object in self.game_info.type_of_objects:
            self.Q_fun[object].eval()
        with torch.no_grad():
            # -- make environment -- #
            env = self.__make_env__(len_game,num_objects,maze,init_pos,start_dist,render_mode='human')

            # -- go through the number of episodes and enact the environment -- #
            for i in range(n_episodes):
                obs, info = env.reset()
                # -- get the objects initial perspective -- #
                for object in self.game_info.type_of_objects:
                    for a in range(num_objects[object]):
                        objects_per[object + f'_{a}'] = [obs[object][f'local_{a}']]
                done = False
                cum_reward={object:[0 for _ in range(num_objects[object])] for object in self.game_info.type_of_objects}
                frame = 0
                # Play
                while not done:
                    # -- get actions -- #
                    actions = self.get_action(env,num_objects,obs,info,epsilon)
                    
                    # -- get next observations -- #
                    next_obs, reward, terminated, truncated, next_info = env.step(actions)
                    
                    for object in self.game_info.type_of_objects:
                        for a in range(num_objects[object]):
                            objects_per[object + f'_{a}'].append(next_obs[object][f'local_{a}'])
                            cum_reward[object][a] += reward[object][a]

                    # -- output frame rewards -- # 
                    if output_frame_rewards:
                        frame+=1
                        print(f'Frame: {frame} reward: {reward}')
                    # -- done -- #
                    done = terminated or truncated
                    obs = next_obs
                    info = next_info
                    self.__last_replay_agents_perspective__ = objects_per
                print(f'cumulative reward: {cum_reward}')
            env.close()
            self.__last_replay_agents_perspective__ = objects_per

    def test_agent(self,maze_dataset,n_episodes,len_game = 50,
                    num_objects = {'agents':1},
                  init_pos = {},
                  start_dist = None):
        """This evaluates how good the agent is at random maze levels
            Returns the ratio completed episodes/total number of episodes""" 
        total_completed = {}
        for obj_type in self.game_info.type_of_objects:
            total_completed[obj_type] = [0 for _ in range(num_objects[obj_type])]
        ep = 0
        # -- go through the number of episodes and enact the environment -- #
        while ep < n_episodes:
            id_x = random.choice(range(len(maze_dataset)))
            maze = maze_dataset[id_x]
            for object in self.game_info.type_of_objects:
                self.Q_fun[object].eval()
            with torch.no_grad():
                # -- make environment -- #
                env = self.__make_env__(len_game,num_objects,maze,init_pos,start_dist)
                
                obs, info = env.reset()

                done = False

                # Play
                while not done:
                    # -- get actions -- #
                    actions = self.get_action(env,num_objects,obs,info)
                    
                    # -- get next observations -- #
                    next_obs, reward, terminated, truncated, next_info = env.step(actions)
                    
                    
                    # -- done -- #
                    done = terminated or truncated
                    obs = next_obs
                    info = next_info
                    
                for obj_type in self.game_info.type_of_objects: 
                    
                    for a in range(num_objects[obj_type]):
                        if info[obj_type + f'_{a}']['success']:
                            total_completed[obj_type][a]+=1/n_episodes
                        
                env.close()
                ep +=1
        for obj_type in self.game_info.type_of_objects:
            total_completed[obj_type] = np.mean(total_completed[obj_type])
        return total_completed
    

    def make_gif(self,name,maze_dataset, len_game = 50, n_episodes = 1, num_objects = {'agents': 1},
                 epsilon = 0,init_pos = {}, start_dist = None,frame_rate=10):
        
        for object in self.game_info.type_of_objects:
            self.Q_fun[object].eval()

        with torch.no_grad():
            maze = maze_dataset[0]
            # -- make environment -- #
            env = self.__make_env__(len_game,num_objects,maze,init_pos,start_dist)
            frames = []
            # -- go through the number of episodes and enact the environment -- #
            for i in range(n_episodes):
                
                random_index = random.randint(0,len(maze_dataset)-1)
                maze = maze_dataset[random_index]

                obs, info = env.reset(options = {'new_maze':maze})
            
                done = False
                
                # Play
                while not done:
                    # -- save render frame -- #
                    frames.append(env.render())

                    # -- get actions -- #
                   
                    actions = self.get_action(env,num_objects,obs,info)
                    
                    # -- get next observations -- #
                    next_obs, reward, terminated, truncated, next_info = env.step(actions)
                    
                    # -- done -- #
                    done = terminated or truncated
                    obs = next_obs
                    info = next_info
                    
            env.close()
        fd = os.getcwd()
        fd = os.path.join(fd, 'media')
        if os.path.exists(fd) == False:
            os.mkdir(fd)
        fd = os.path.join(fd,f'{name}.gif')  
        imageio.mimsave(fd, frames, fps=frame_rate,loop=0)
               

    def animate_last_replay(self,type_object,object_id,name, save = False):
        """Takes the last run_agent and saved perspectives of the agents and 
            returns a animation of it
            agent_id: gives the id of the agent for the perspective we want
            name: name of the replay to save in the media folder"""
        
        seq_anim = self.__last_replay_agents_perspective__[type_object + f'_{object_id}']
        html, ani = create_animation(seq_anim)
        display(html)
        if save:
            fd = os.getcwd()
            fd = os.path.join(fd, 'media')
            if os.path.exists(fd) == False:
                os.mkdir(fd)
            fd = os.path.join(fd,f'{name}.gif')
            ani.save(fd, writer='pillow')

    def __getModelparam__(self,type_object): 

            model_param = {
            'model_name': self.Q_fun[type_object].name,
            'vision': self.vision[type_object],
            'action_type': self.action_type,
            'dist_paradigm':self.dist_paradigm,
            'game_name': self.game_info.name
            }        
            return model_param

    def save(self,name):
        """Save the agents models"""
        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')

        if os.path.exists(fd)==False:
            os.mkdir(fd)

        fd = os.path.join(fd,f'{name}')
        if os.path.exists(fd)==False:
            os.mkdir(fd)
        for object in self.game_info.type_of_objects:
            # -- save the agent model -- #
            object_fd = os.path.join(fd,object)
            if os.path.exists(object_fd)==False:
                os.mkdir(object_fd)
            torch.save(self.Q_fun[object].state_dict(),os.path.join(object_fd,f'agent.pth'))

            object_model_param = self.__getModelparam__(object)

            # -- save the type of model with other agent specific parameters -- #
            with open(os.path.join(object_fd,'model_hyperparameters.json'),'w') as f:
                json.dump(object_model_param, f, indent=4)
        
            reward_structure = self.game_info.rewards_dist[object].rewards
            reward_structure['WALL'] = Maze_env.env.mazes.WALL
            reward_structure['DO_ACTION'] = Maze_env.env.mazes.DO_ACTION
        
            # -- save reward distribution -- #
            with open(os.path.join(object_fd,'reward_distribution.json'),'w') as f:
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