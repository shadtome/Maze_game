import numpy as np
import matplotlib.pyplot as plt
import maze_dataset.maze.lattice_maze as lm
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot
import random
import gymnasium as gym
from matplotlib.colors import Normalize
import pygame
import torch

# -- Reward parameters -- #
DO_ACTION = -0.1
WALL = -0.75
STAY = 0.0

class BasicMaze(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'],'render_fps':12,
                'obs_type': ['rgb','spatial','basic'],
                'action_type': ['full','cardinal'],
                'dist_paradigm' : ['radius','path']}
    
    def __init__(self, maze, len_game=1000,num_objects={'agents': 1},
                 vision_len = {'agents': 1}, type_of_objects = ['agents'],
                 objectives = {'agents': 'goal'},
                 action_type = 'full',render_mode = None, obs_type=None,
                 init_pos={}, start_dist = None,
                 dist_paradigm = 'radius', collision_rules = None,
                 colormap = {'agents': 'tab10'}):
        
        """ Maze Runner environment:

            maze: inputed maze 

            len_game: the total length of the game

            num_agents: number of agents in the environment

            vision_len: the length of vision the agents can see

            action_type: either 'full' for the cardinal directions and stop actions, 
                        while 'cardinal' is just the cardinal directions.

            render_mode: 'rgb_array' for training, while 'human' will output a pygame instance
            obs_type: either 'spatial' for local rgb colored images around the agents, or 'basic' 
                        for just positions of agents and goals.

            agents_pos: the environment randomly selects the agents positions if this is None, 
                        otherwise you can input a list of the agents position

            targets_pos: the environment randomly selects the targets positions, if this is None,
                        otherwise,, you can input a list of the target's position"""
        super(BasicMaze, self).__init__()

        # -- team colors -- #
        self.team_colors = ["Greys","Blues", "Reds", "Greens", "Purples", "Oranges"]
        
        # --- window size for pygame output --- #
        self.window_size = 512 

        # -- type of non-goal objects in the maze -- #
        self.type_of_objects = type_of_objects

        # --- maze and positions of agents and targets --- #
        self.maze = maze
        self.init_pos = init_pos

        # -- collision on or off -- #
        self.collision_rules = collision_rules

        # -- type of randominzation of the goal with respect to the agent -- #
        self.dist_paradigm=dist_paradigm

        # -- set up color map for objects -- #
        self.colormap = colormap
        
        
        # -- maze information -- #
        self.n_cols = maze.grid_shape[1]
        self.n_rows = maze.grid_shape[0]
        self.min_size = min(self.n_cols,self.n_rows)
        self.max_dist = self.__max_dist__()
        
        # -- game information -- #
        self.objectives = objectives
        self.pos = {}
        self.done = {}
        self.len_game = len_game
        self.vision_len = vision_len
        self.num_objects = num_objects
        self.dead = {}
        self.success = {}
        self.path = {}

        # -- game object mappings -- #
        self.vision_mapping = {'nothing': 0 , 'wall': 1, 'other': 2, 'goal':3}
        

        # -- start distance -- #
        self.start_dist = start_dist
        if self.start_dist==None:
            if self.dist_paradigm == 'radius':
                self.start_dist = self.n_cols + self.n_rows - 2
            else:
                self.start_dist = self.n_cols*self.n_rows - 1
        else:
            # check if the distance is allowable:
            if self.start_dist <=0:
                raise ValueError(f'start_dist must be greater then 0')

        # -- observation space type -- #
        assert obs_type is None or obs_type in self.metadata['obs_type']
        self.obs_type = obs_type

        # -- action space type -- #
        assert action_type is None or action_type in self.metadata['action_type']
        self.action_type = action_type

        if action_type == 'full':
            self.action_space = gym.spaces.Discrete(5)
            self.actions = ['UP','DOWN','LEFT','RIGHT','STAY']
        elif action_type == 'cardinal':
            self.action_space = gym.spaces.Discrete(4)
            self.actions = ['UP','DOWN','LEFT','RIGHT']

        self.agent_target_obs_space = gym.spaces.Discrete(self.n_cols*self.n_rows)

        # -- observation space -- #
        self.observation_space = None
        if self.obs_type == 'basic':

            obs = dict()
            for a in range(self.num_objects['agents']):
                # We set up the observation spaces as discrete: pos = n_cols * row + col
                obs[f'agent_{a}'] = gym.spaces.Discrete(self.n_cols*self.n_rows)
                obs[f'target_{a}']= gym.spaces.Discrete(self.n_cols*self.n_rows)
            self.observation_space = gym.spaces.Dict(obs)

        elif self.obs_type == 'rgb':
            
            self.observation_space = gym.spaces.Box(0,255,shape=(self.window_size,self.window_size,3))
            
        
        elif self.obs_type == 'spatial':
            obs = dict()
            max_pos = self.n_rows*self.n_cols-1
            for obj in self.type_of_objects:
                obs_objects = dict()
                for a in range(self.num_objects[obj]):
                    
                    obs_objects[f'local_{a}'] = gym.spaces.Box(low=0,
                                                    high=255,
                                                    shape = (2*self.vision_len[obj] + 1,2*self.vision_len[obj]+1,3),
                                                    dtype=int)
                    obs_objects[f'global_{a}'] = gym.spaces.Box(low=np.array([0,0,0,0]),
                                                        high=np.array([max_pos,max_pos,1,self.n_cols + self.n_rows]),
                                                        shape=(4,),
                                                        dtype=int)
                obs[obj] = gym.spaces.Dict(obs_objects)
            self.observation_space = gym.spaces.Dict(obs)
        

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.timer = 0
        self.window = None
        self.clock = None

    def __max_dist__(self):
        """ Maximum distance between two points in the maze based on the distance paradigm"""
        if self.dist_paradigm == 'radius':
            return self.n_cols + self.n_rows - 2
        if self.dist_paradigm == 'path':
            return self.n_cols*self.n_rows - 1

    def new_maze(self,maze):
        """ Used to give the environment a new maze design"""
      
        self.__init__(maze,self.len_game,self.num_objects,
                      self.vision_len,self.type_of_objects,self.objectives,self.action_type,self.render_mode,
                      self.obs_type,
                      init_pos = self.init_pos, dist_paradigm=self.dist_paradigm,
                      collision_rules=self.collision_rules)
        

    def __global__(self,a,type_object):
        """ gets the global information for the agent, which includes the position, goal,"""
        global_var = []
        pos = self.pos[type_object][a]
        if type_object + '_goals' in self.pos:
            g_pos = self.pos[type_object + '_goals'][a]
        else:
            g_pos=-1
        global_var.append(pos)
        global_var.append(g_pos)
        global_var.append(self.done[type_object][a])
        global_var.append(self.manhattan_dist(pos,g_pos))
        return np.array(global_var)
    
    def __getCoords__(self,pos):
        """ gets the coordinates of the position"""
        x_pos = pos % self.n_cols
        y_pos = pos // self.n_cols
        return x_pos,y_pos

    def __place_value__(self,val,alpha,beta,diag = False):
    
            if val == self.vision_mapping['nothing'] and diag == False:
                return [int(255* beta),int(255* beta),int(255* beta)]
            elif val == self.vision_mapping['other'] and diag == False:
                return [int(255* beta),0,0]

            if val == self.vision_mapping['nothing'] and diag:
                return [int(255*alpha*beta),int(255*alpha*beta),int(255*alpha*beta)]
            elif val == self.vision_mapping['other'] and diag:
                return [int(255*alpha*beta),0,0]
            else:
                return [0,0,0]
            
    def __place_center_value__(self,val):
        """ places the center value of the agent"""
        if val == self.vision_mapping['nothing']:
            return [0,0,255]
        elif val == self.vision_mapping['other']:
            return [255,0,0]
        else:
            return [0,0,0]
        
    def __spatial__(self, a,type_object):
        """ gets the spatial information for the agent, which includes the local rgb information"""
        
        grid_size = 2*self.vision_len[type_object] + 1
        
        vision = self.__get_vision__(a,type_object)
        spatial = np.zeros(shape=(grid_size,grid_size,3),dtype=int)
        x = self.vision_len[type_object]
        y = self.vision_len[type_object]

        alpha = 0.3
        beta = 0.99
        
        # CENTER 
        spatial[y][x] = self.__place_center_value__(vision['CENTER'][0])
        
        # UP 
        for i,val in enumerate(vision['UP']):
            spatial[y-(i+1)][x] = self.__place_value__(val,alpha,beta/(i+1))
        
        # DOWN 
        for i,val in enumerate(vision['DOWN']):
            spatial[y+(i+1)][x] = self.__place_value__(val,alpha,beta/(i+1))

        # LEFT 
        for i,val in enumerate(vision['LEFT']):
            spatial[y][x-(i+1)] = self.__place_value__(val,alpha,beta/(i+1))

        # RIGHT
        for i,val in enumerate(vision['RIGHT']):
            spatial[y][x + (i+1)] = self.__place_value__(val,alpha,beta/(i+1))

        # UP LEFT
        for i,val in enumerate(vision['UP_LEFT']):
            spatial[y-(i+1)][x-(i+1)] = self.__place_value__(val,alpha,beta/(i+1),True)
        
        # UP RIGHT
        for i,val in enumerate(vision['UP_RIGHT']):
            spatial[y-(i+1)][x+(i+1)] = self.__place_value__(val,alpha,beta/(i+1),True)

        # DOWN LEFT
        for i,val in enumerate(vision['DOWN_LEFT']):
            spatial[y+(i+1)][x-(i+1)] = self.__place_value__(val,alpha,beta/(i+1),True)

        # DOWN RIGHT
        for i,val in enumerate(vision['DOWN_RIGHT']):
            spatial[y+(i+1)][x+(i+1)] = self.__place_value__(val,alpha,beta/(i+1),True)
        
            
        return spatial      

    def __localRegion__(self):
        """Not used!"""
        # first lets get our full rgb image and pad it out 
        pix_square_size = (
            self.window_size//self.min_size
        )
        pixels = self.__render_frame__()
        pixels = torch.tensor(pixels)
        vision = self.vision_len['agents']
        vision_ext = int(vision*pix_square_size)
        padded_pixels = torch.nn.functional.pad(pixels.permute(2,0,1),
                                                pad=(vision_ext,vision_ext,vision_ext,vision_ext),
                                                mode='constant',value=0)
        local = []
        for a in range(self.num_objects['agents']):
            x,y = self.__getCoords__(self.pos['agents'][a])
            point = np.array([x,y],dtype=int) + vision
            
            center = pix_square_size * point + pix_square_size//2
            x = int(center[0])
            y = int(center[1])
            region = padded_pixels[ :, y-vision_ext-pix_square_size//2:y+vision_ext + pix_square_size//2, 
                                   x-vision_ext - pix_square_size//2:x+vision_ext + pix_square_size//2]
            local.append(region.permute(1,2,0))
        return local
          
    def __get_path__(self,pos,t_pos):
        """ gets the path from the position to the target position"""
        x ,y = self.__getCoords__(pos)
        x_t,y_t = self.__getCoords__(t_pos)

        # -- get path from their initial location to their target -- #
        agents_path = self.maze.find_shortest_path(c_start = (y,x),
                                                c_end = (y_t,x_t))
        transformed = []
        for row, col in agents_path:
            transformed.append(self.n_cols * row + col)
        return transformed

    def __update_init_pos__(self,init_pos):
        """ updates the initial positions of the objects"""
        self.init_pos = init_pos

    def __update_start_dist__(self,start_dist=None):
        """ updates the start distance"""
        self.start_dist = start_dist
        if self.start_dist == None:
            if self.dist_paradigm == 'radius':
                self.start_dist = self.n_cols + self.n_rows -2
            else:
                self.start_dist = self.n_cols * self.n_rows -1

    def __get_unique_pos__(self,type_object,pos_set,index):
        """ gets a unique position for the object"""
        if type_object not in self.init_pos:
            pos = self.agent_target_obs_space.sample()
            while pos in pos_set :
                pos = self.agent_target_obs_space.sample()
            pos_set.add(pos)
            return pos
        else:
            return self.init_pos[type_object][index]
        
    def __init_object_pos__(self,type_object,pos_set):
        object_positions = []
        object_done = []
        dead = []
        success = []

        if type_object in self.init_pos:
            pos_set = pos_set | set(self.init_pos[type_object])

        # -- go through each object and initalize the the positions if they have not been initalized -- #
        for a in range(self.num_objects[type_object]):

            # -- first lets get our agent positions -- #
            pos = self.__get_unique_pos__(type_object,pos_set,a)
            object_positions.append(pos)

            # -- append False for the agent is done parameter -- #
            object_done.append(False)
            dead.append(False)
            success.append(False)

        self.pos[type_object] = object_positions
        self.done[type_object] = object_done
        self.dead[type_object] = dead
        self.success[type_object] = success

    def __init_individual_objects__(self,type_object,pos_set):
        self.__init_object_pos__(type_object,pos_set)
    """
    def __init_individual_objects__(self,type_object,pos_set):
        
        has_goal = False
        if self.objectives[type_object] == 'goal':
            object_goals = []
            object_path = []
            has_goal = True
        
        object_positions = []
        object_done = []
        dead = []
        success = []

        # -- first lets put in our positions that are initialized by the user -- #
        if type_object in self.init_pos:
            pos_set = pos_set | set(self.init_pos[type_object])
        
        if type_object+ '_goals' in self.init_pos:
            pos_set = pos_set | set(self.init_pos[type_object + '_goals'])

        # -- go through each object and initalize the the positions if they have not been initalized -- #
        for a in range(self.num_objects[type_object]):

            # -- first lets get our agent positions -- #
            pos = self.__get_unique_pos__(type_object,pos_set,a)
            object_positions.append(pos)

            # -- second lets get our target positions for the agent
            if has_goal:
                t_pos = self.__get_unique_pos__(type_object + '_goals',pos_set,a)
                object_goals.append(t_pos)

            # -- append False for the agent is done parameter -- #
            object_done.append(False)
            dead.append(False)
            success.append(False)

            # -- get minimal path to goal -- #
            if has_goal:
                object_path.append(self.__get_path__(pos,t_pos))
                
        
        self.pos[type_object] = object_positions
        if has_goal:
            self.pos[type_object+'_goals'] = object_goals
            self.path[type_object] = object_path
        self.done[type_object] = object_done
        self.dead[type_object] = dead
        self.success[type_object] = success
    """
    def __init_objects__(self):
        """ Initialize the agents: this includes finding their initial positions,
         the initial target positions, and subsequent information like being done, 
          or the path """
        pos_set = set()
        for object_type in self.type_of_objects:
            self.__init_individual_objects__(object_type,pos_set)
        
        
    def get_dist(self,point1,point2):
        """ gets the distance between two points based on the distance paradigm"""
        if self.dist_paradigm == 'radius':
            return self.manhattan_dist(point1,point2)
        
        if self.dist_paradigm == 'path':
            return len(self.__get_path__(point1,point2)) - 1

    def manhattan_dist(self,point1, point2):
        x_1,y_1 = self.__getCoords__(point1)
        x_2,y_2 = self.__getCoords__(point2)
        return abs(x_1 - x_2) + abs(y_1 - y_2)

    def __what_they_see__(self,pos,pos_set,type_object,a):
        if pos in pos_set:
            return self.vision_mapping['other']
        else:
            return self.vision_mapping['nothing']

    def __get_vision__(self,a,type_object):
        """ gets the vision of the agent"""

        down_con_list = self.maze.connection_list[0]
        right_con_list = self.maze.connection_list[1]
        

        # Flatten so that the connection list is a list of bool
        down_con_list = np.array(down_con_list).flatten().tolist()
        right_con_list = np.array(right_con_list).flatten().tolist()
        
        n_rows = self.n_rows
        n_cols = self.n_cols
        # set up the boundaries.
        # top edge: pos <n_cols
        # bot edge: pos >= n_cols*(n_rows-1)
        # left edge: pos % n_cols == 0
        # right edge: pos % n_cols == n_cols-1

        # Furthermore, if pos is the position
        # going down: pos + n_cols
        # going up: pos - n_cols
        # going right: pos + 1
        # going left: pos - 1
        
        vision = {}
        positions = self.pos.copy()
        
        pos = positions[type_object][a]
        
        # agents that are not done
        not_done = {}
        for obj in self.type_of_objects:
            not_done[obj] = np.where(np.array(self.done[obj]) == False)[0]
        
        
        pos_set = set()
        for obj in self.type_of_objects:
            pos_set = pos_set | {positions[obj][i] for i in not_done[obj] }

        # CENTER 
        center_vision = []
        other_pos = pos_set - {pos}
        center_vision.append(self.__what_they_see__(pos,other_pos,type_object,a))
        vision['CENTER'] = center_vision
        
        # UP
        up_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = (pos_temp//n_cols<=0 or down_con_list[pos_temp - n_cols] == False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                up_vision += rest_wall
                break
            pos_temp -= n_cols
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                up_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['UP'] = up_vision

        # DOWN 
        down_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = (pos_temp//n_cols>=n_rows-1 or down_con_list[pos_temp]==False )
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                down_vision += rest_wall
                break
            pos_temp += n_cols
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                down_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['DOWN'] = down_vision

        # LEFT 
        left_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = (pos_temp % n_cols <= 0 or right_con_list[pos_temp - 1]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                left_vision += rest_wall
                break
            pos_temp -= 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                left_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['LEFT'] = left_vision

        # RIGHT
        right_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = (pos_temp % n_cols >= n_cols-1 or right_con_list[pos_temp]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                right_vision += rest_wall
                break
            pos_temp +=1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                right_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['RIGHT'] = right_vision

        # UP LEFT
        up_left_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = ((pos_temp % n_cols <= 0 or pos_temp//n_cols<=0) or
                  (right_con_list[pos_temp-n_cols-1]==False or down_con_list[pos_temp-n_cols]==False)
                   and (down_con_list[pos_temp-n_cols-1]==False or right_con_list[pos_temp - 1]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                up_left_vision += rest_wall
                break
            pos_temp = pos_temp -1 - n_cols
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                up_left_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['UP_LEFT'] = up_left_vision

        # UP RIGHT
        up_right_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = ((pos_temp % n_cols >= n_cols-1 or pos_temp//n_cols<=0) or
                  (right_con_list[pos_temp-n_cols]==False or down_con_list[pos_temp-n_cols]==False)
                   and (down_con_list[pos_temp-n_cols+1]==False or right_con_list[pos_temp]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                up_right_vision += rest_wall
                break
            pos_temp = pos_temp - n_cols +1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                up_right_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['UP_RIGHT'] = up_right_vision

        # DOWN LEFT
        down_left_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = ((pos_temp % n_cols <= 0 or pos_temp//n_cols>=n_rows-1) or
                  (right_con_list[pos_temp-1]==False or down_con_list[pos_temp-1]==False)
                   and (down_con_list[pos_temp]==False or right_con_list[pos_temp+n_cols -1]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                down_left_vision += rest_wall
                break
            pos_temp  = pos_temp +n_cols -1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                down_left_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['DOWN_LEFT'] = down_left_vision

        # DOWN RIGHT
        down_right_vision = []
        pos_temp = pos
        for i in range(self.vision_len[type_object]):
            
            v = ((pos_temp % n_cols >= n_cols-1 or pos_temp//n_cols>=n_rows-1) or
                  (right_con_list[pos_temp]==False or down_con_list[pos_temp+1]==False)
                   and (down_con_list[pos_temp]==False or right_con_list[pos_temp+n_cols]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len[type_object] - i)]
                down_right_vision += rest_wall
                break
            pos_temp = pos_temp + n_cols + 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                down_right_vision.append(self.__what_they_see__(pos_temp,pos_set,type_object,a))
            
        vision['DOWN_RIGHT'] = down_right_vision

        return vision
        #return {
            #'UP': y == 0 or self.maze.connection_list[0][y-1][x] == False,
            #'DOWN': y == self.maze_shape[0]-1 or self.maze.connection_list[0][y][x] == False,
            #'LEFT': x == 0 or self.maze.connection_list[1][y][x-1] == False,
            #'RIGHT': x == self.maze_shape[1] - 1 or self.maze.connection_list[1][y][x] == False
        #}

    def reset(self,seed=None,options=None):
        """ resets the environment"""
        if options!=None:
            if 'new_maze' in options:
                self.new_maze(options['new_maze'])
            if 'init_pos' in options:
                self.__update_init_pos__(options['init_pos'],None)
            if 'start_dist' in options:
                self.__update_start_dist__(options['start_dist'])
            return self.reset()
        else:
            super().reset(seed=seed)
            self.timer = 0
            self.__init_objects__()
            state = {}
            for type_object in self.type_of_objects:
                state[type_object] = self.__get_state__(type_object)
            info = self.__get_info__()
            
            if self.render_mode == 'human':
                self.__render_frame__()
            
            return state, info
        
    def __get_state__(self,type_object):

        if self.obs_type == 'basic':
            state = {}
            for a in range(self.num_objects[type_object]):
                state[type_object + f'_{a}'] = self.pos[type_object][a]
                if type_object + '_goals' in self.pos:
                    state[f'target_{a}'] = self.pos[type_object+'_goals'][a]
            return state
        elif self.obs_type == 'rgb':
            # Note that this is in (window,window,3)
            return self.__render_frame__()
        elif self.obs_type == 'spatial':
            state = {}
            for a in range(self.num_objects[type_object]):
                state[f'local_{a}'] = self.__spatial__(a,type_object)
                state[f'global_{a}'] = self.__global__(a,type_object)
                
            return state
        
    def __get_object_info__(self,type_object):
        info = {}
        for a in range(self.num_objects[type_object]):
            pos = self.pos[type_object][a]
            
            vision = self.__get_vision__(a,type_object)
            info[type_object + f'_{a}'] = {'CENTER_vision': vision['CENTER'],
                                  'UP_vision' : vision['UP'],
                                  'DOWN_vision': vision['DOWN'],
                                  'LEFT_vision': vision['LEFT'],
                                  'RIGHT_vision': vision['RIGHT'],
                                  'UP_LEFT_vision': vision['UP_LEFT'],
                                  'UP_RIGHT_vision': vision['UP_RIGHT'],
                                  'DOWN_LEFT_vision': vision['DOWN_LEFT'],
                                  'DOWN_RIGHT_vision': vision['DOWN_RIGHT'],
                                  'done' : self.done[type_object][a],
                                  'dead' : self.dead[type_object][a],
                                  'success' : self.success[type_object][a]}

            if self.obs_type == 'rgb' or self.obs_type=='spatial':
                info[type_object + f'_{a}']['pos'] = pos
        info[type_object] = {}
        return info
 
    def __get_info__(self):
        info = {}
        for type_object in self.type_of_objects:
            info = info | self.__get_object_info__(type_object)
            info['n_'+type_object] = self.num_objects[type_object]
        info['timer']=self.timer
        info['len_game'] = self.len_game
        info['max_dist'] = self.max_dist
        info['max_pos'] = self.n_cols*(self.n_rows) - 1
        info['type_of_objects'] = self.type_of_objects
        
        return info
    
    def __collisions__(self,prev_pos,cur_pos,type_object_1,type_object_2):
        """ Check if there are any collisions between the agents"""
        """Efficient collision detection between two object types."""
        collisions = []
        cur_pos_dict = {}
        path_dict = {}

        # Helper: add to dictionaries
        def add_object(obj_type, i):
            p_pos = prev_pos[obj_type][i]
            c_pos = cur_pos[obj_type][i]
            obj_info = {'type': obj_type, 'index': i, 'prev': p_pos, 'cur': c_pos}

            # Group by current position for same-position detection
            cur_pos_dict.setdefault(c_pos, []).append(obj_info)

            # Store path for quick swap detection
            path_dict[(p_pos, c_pos)] = obj_info

        # Add all objects that are not done
        for i in range(self.num_objects[type_object_1]):
            if self.done[type_object_1][i]==False:
                add_object(type_object_1, i)
        for i in range(self.num_objects[type_object_2]):
            if self.done[type_object_2][i] == False:
                add_object(type_object_2, i)

        # Detect same-position collisions
        for position, objs in cur_pos_dict.items():
            if len(objs) > 1:
                for i in range(len(objs)):
                    for j in range(i + 1, len(objs)):
                        if (objs[i]['type'], objs[i]['index']) != (objs[j]['type'], objs[j]['index']):
                            collisions.append({
                                'object_1': objs[i],
                                'object_2': objs[j],
                                'collision_type': 'same_position'
                            })

        # Detect swap collisions
        checked_swaps = set()
        for (p_pos, c_pos), obj in path_dict.items():
            if (c_pos, p_pos) in path_dict and (c_pos, p_pos) not in checked_swaps:
                obj2 = path_dict[(c_pos, p_pos)]
                # Prevent double counting
                
                if (obj['type'],obj['index'])!=(obj2['type'],obj2['index']):
                    checked_swaps.add((p_pos, c_pos))
                    checked_swaps.add((c_pos, p_pos))
                    collisions.append({
                        'object_1': obj,
                        'object_2': obj2,
                        'collision_type': 'swap'
                    })

        return collisions
    
    def __results_collisions__(self,prev_pos):
        """ Detects collisions between the agents
            collision_rules = {(object_1, object_2): {object_1: 'die', object_2: 'die'}}"""
        
        if self.collision_rules !=None:
            for objects,rules in self.collision_rules.items():
                collisions = self.__collisions__(prev_pos,self.pos,objects[0],objects[1])
                
                for collision in collisions:
                    if rules[objects[0]] == 'die':
                        self.dead[objects[0]][collision['object_1']['index']] = True
                        self.done[objects[0]][collision['object_1']['index']] = True
                    if rules[objects[1]] == 'die':
                        self.dead[objects[1]][collision['object_2']['index']] = True
                        self.done[objects[1]][collision['object_2']['index']] = True

    def __is_done__(self,type_object,pos,index):
        """ checks if the agent is done based on a variety of conditions\n
        Can be easily extended through inheritence"""
        # -- if the object has a end goal check if they reached it -- #
        return 
        
        


    def __object_step__(self,type_object,actions):
        """ steps through the environment for the object"""
        # Furthermore, if pos is the position
        # going down: pos + n_cols
        # going up: pos - n_cols
        # going right: pos + 1
        # going left: pos - 1

        # --- get number of columns and rows --- #
        n_cols = self.n_cols

        object_rewards = []
        
        # --- go through each agent's action and process it --- #
        for i, action in enumerate(actions):

            rewards = 0
            pos = self.pos[type_object][i]
            vision = self.__get_vision__(i,type_object)

            # --- move the agent is the corresponding direction if there is not wall --- #
            if action == 0 and vision['UP'][0]!=1:
                
                rewards +=DO_ACTION
                pos -= n_cols
            elif action == 1 and vision['DOWN'][0]!=1:
                
                rewards +=DO_ACTION
                pos += n_cols
            elif action == 2 and vision['LEFT'][0]!=1:
                
                rewards +=DO_ACTION
                pos -= 1
            elif action == 3 and vision['RIGHT'][0]!=1:
                
                rewards +=DO_ACTION
                pos += 1
            else:
                if action !=4:
                    # --- penalize for going against the wall --- #
                    
                    rewards +=WALL
                rewards +=STAY
            
            # --- determine if agent is done or not --- #
            self.__is_done__(type_object,pos,i)
            
            # -- if agent is not done, update position -- #
            if self.done[type_object][i] == False:
                self.pos[type_object][i] = pos


            object_rewards.append(rewards)
      
        # --- check if all agents are done --- #
        done = all(self.done[type_object])

        # --- check if end of the time --- #
        truncated = self.timer == self.len_game

        if self.render_mode == 'human' and self.obs_type!='rgb':
            pixels = self.__render_frame__()
        
        return self.__get_state__(type_object),np.array(object_rewards), done, truncated
    
    def step(self,actions):
        # -- data to consolidate -- #
        object_states = {}
        object_rewards = {}
        all_done = True
        all_truncated = True
        # -- increment the timer -- #
        self.timer+=1
        # -- save the previous position -- #
        prev_pos = self.pos.copy()
        # -- go through each type of object and step through the environment -- #
        for object in self.type_of_objects:
            state, rewards, done, truncated = self.__object_step__(object,actions[object])
            object_states[object] = state
            object_rewards[object] = rewards
            all_done = all_done and done
            all_truncated = all_truncated and truncated

        # -- check for collisions -- #
        self.__results_collisions__(prev_pos)
        
        return object_states, object_rewards, all_done, all_truncated, self.__get_info__()

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.__render_frame__()

    def __render_frame__(self):
        
        
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Maze Runners')
            self.window = pygame.display.set_mode(
                (self.window_size,self.window_size)
            )


        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()



        screen = pygame.Surface((self.window_size, self.window_size))
        screen.fill((255,255,255))
        pix_square_size = (
            self.window_size/self.min_size
        )

        # Draw the targets:
        for type_object in self.type_of_objects:
            self.__add_to_env__(type_object,screen,pix_square_size)

        if self.render_mode == 'human':
            self.window.blit(screen,screen.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)),axes = (1,0,2)
        )

    def __draw_maze__(self,screen, square_size):
        # --- get connection list for the maze --- #
        # --- down connection: true if go down, right connection: true if go right --- #
        down_con = self.maze.connection_list[0]
        right_con = self.maze.connection_list[1]
        rows = self.n_rows
        cols = self.n_cols
        # --- go through each of the rows and columns to see if there is a wall and draw it --- #
        for row in range(rows):
            for col in range(cols):

                x = col * square_size
                y = row * square_size

                # --- draw line for column --- #
                if col < cols - 1 and not right_con[row][col]:
                    pygame.draw.line(screen, (0,0,0), (x + square_size, y),
                                      (x + square_size, y + square_size),5)
                # --- draw line for row --- #
                if row < rows - 1 and not down_con[row][col]:
                    pygame.draw.line(screen, color=(0,0,0), start_pos=(x, y+square_size),
                                     end_pos=(x+square_size,y+square_size),width=5)
                    
    def __add_object_to_env__(self,type_object,screen,pix_square_size):
        norm = Normalize(vmin=0,vmax=self.num_objects[type_object])
        colormap = plt.cm.get_cmap(self.colormap[type_object],self.num_objects[type_object])
        # Draw the agents
        for i, point in enumerate(self.pos[type_object]):
            if self.done[type_object][i]==False:
                color = self.__get_color__(colormap,norm,i,type_object)
                x,y = self.__getCoords__(point)
                pos = np.array([x,y])
                pygame.draw.circle(
                    surface = screen,
                    color = tuple(int(c*255) for c in color[:3]),
                    center = (pos + 0.5) * pix_square_size,
                    radius = pix_square_size/3,
                )
    def __add_to_env__(self,type_object,screen,pix_square_size):
        self.__draw_maze__(screen,pix_square_size)
        self.__add_object_to_env__(type_object,screen,pix_square_size)

    def __get_color__(self,colormap,norm,index,type_object):
        
        if self.colormap[type_object] in self.team_colors:
            return colormap(1)
        else:
            return colormap(norm(index))
        
                            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
                    
                


