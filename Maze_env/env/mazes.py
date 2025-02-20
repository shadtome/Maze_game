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

# We will set up our state space as ((x,y))
# This is just the position for now

# 0 stay
# 1 move up
# 2 move down
# 3 move left
# 4 move right


# -- Reward parameters -- #
DO_ACTION = 0.0
WALL = -0.75
STAY = 0.0

class maze_env(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'],'render_fps':4,
                'obs_type': ['rgb','spatial','basic'],
                'action_type': ['full','cardinal']}
    
    def __init__(self, maze, len_game=1000,num_agents=1,vision_len = 1, 
                 action_type = 'full',render_mode = None, obs_type=None,
                 agents_pos= None, targets_pos = None):
        
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
        super(maze_env, self).__init__()

        
        # --- window size for pygame output --- #
        self.window_size = 512 

        # --- maze and positions of agents and targets --- #
        self.maze = maze
        self.init_pos = dict()
        self.init_pos['agents'] = agents_pos
        self.init_pos['targets'] = targets_pos
        
        # -- maze information -- #
        self.n_cols = maze.grid_shape[1]
        self.n_rows = maze.grid_shape[0]
        self.min_size = min(self.n_cols,self.n_rows)

        self.len_game = len_game
        self.vision_len = vision_len
        self.num_agents = num_agents
        self.agent_positions = None
        self.agent_goals = None
        self.agents_done = None
        self.agents_path = None
        

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
            for a in range(self.num_agents):
                # We set up the observation spaces as discrete: pos = n_cols * row + col
                obs[f'agent_{a}'] = gym.spaces.Discrete(self.n_cols*self.n_rows)
                obs[f'target_{a}']= gym.spaces.Discrete(self.n_cols*self.n_rows)
            self.observation_space = gym.spaces.Dict(obs)

        elif self.obs_type == 'rgb':
            
            self.observation_space = gym.spaces.Box(0,255,shape=(self.window_size,self.window_size,3))
            
        
        elif self.obs_type == 'spatial':
            obs = dict()
            max_pos = self.n_rows*self.n_cols-1
            for a in range(self.num_agents):
                
                obs[f'local_{a}'] = gym.spaces.Box(low=0,
                                                   high=255,
                                                   shape = (2*self.vision_len + 1,2*self.vision_len+1,3),
                                                   dtype=int)
                obs[f'global_{a}'] = gym.spaces.Box(low=np.array([0,0,0,0]),
                                                    high=np.array([max_pos,max_pos,1,self.n_cols + self.n_rows]),
                                                    shape=(4,),
                                                    dtype=int)

            self.observation_space = gym.spaces.Dict(obs)
        


        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.timer = 0
        self.window = None
        self.clock = None

    def new_maze(self,maze):
        """ Used to give the environment a new maze design"""
        self.__init__(maze,self.len_game,self.num_agents,
                      self.vision_len,self.action_type,self.render_mode,
                      self.obs_type,
                      agents_pos = self.init_pos['agents'],
                      targets_pos=self.init_pos['targets'])

    def __global__(self,a):
        """ Get global stats for the state 
            a: agent index """
        global_var = []
        pos = self.agent_positions[a]
        g_pos = self.agent_goals[a]
        global_var.append(pos)
        global_var.append(g_pos)
        global_var.append(self.agents_done[a])
        global_var.append(self.manhattan_dist(pos,g_pos))
        return np.array(global_var)

    def __spatial__(self, a):
        """ gets local spatial information, i.e. get the local image data for the 
            agent, which includes the walls, other agents, their goals and 
            it has a weight to measure the distance from it."""
        
        grid_size = 2*self.vision_len + 1
        
        vision = self._get_vision(a)
        spatial = np.zeros(shape=(grid_size,grid_size,3),dtype=int)
        x = self.vision_len
        y = self.vision_len
        spatial[y][x][0] = 0
        spatial[y][x][1] = 0
        spatial[y][x][2] = 255

        def place_value(val,x,y,dist,diag = False):
            alpha = 0.3
            beta = 0.99/dist
            if val == 0 and diag == False:
                spatial[y][x][0] = int(255* beta)
                spatial[y][x][1] = int(255* beta)
                spatial[y][x][2] = int(255* beta)
            elif val == 2 and diag == False:
                spatial[y][x][0] = int(255* beta)
                spatial[y][x][1] = 0
                spatial[y][x][2] = 0
            elif val == 3 and diag == False:
                spatial[y][x][0] = 0
                spatial[y][x][1] = int(255* beta)
                spatial[y][x][2] = 0

            if val == 0 and diag:
                spatial[y][x][0] = int(255*alpha*beta)
                spatial[y][x][1] = int(255*alpha*beta)
                spatial[y][x][2] = int(255*alpha*beta)
            elif val == 2 and diag:
                spatial[y][x][0] = int(255*alpha*beta)
                spatial[y][x][1] = 0
                spatial[y][x][2] = 0
            elif val == 3 and diag:
                spatial[y][x][0] = 0
                spatial[y][x][1] = int(255*alpha*beta)
                spatial[y][x][2] = 0
        
        # UP 
        for i,val in enumerate(vision['UP']):
            place_value(val,x,y-(i+1),i+1)
        
        # DOWN 
        for i,val in enumerate(vision['DOWN']):
            place_value(val,x,y+(i+1),i+1)

        # LEFT 
        for i,val in enumerate(vision['LEFT']):
            place_value(val,x - (i+1),y,i+1)

        # RIGHT
        for i,val in enumerate(vision['RIGHT']):
            place_value(val,x + (i+1),y,i+1)

        # UP LEFT
        for i,val in enumerate(vision['UP_LEFT']):
            place_value(val, x-(i+1),y-(i+1),i+1,True)
        
        # UP RIGHT
        for i,val in enumerate(vision['UP_RIGHT']):
            place_value(val, x+(i+1),y-(i+1),i+1,True)

        # DOWN LEFT
        for i,val in enumerate(vision['DOWN_LEFT']):
            place_value(val, x-(i+1),y+(i+1),i+1,True)

        # DOWN RIGHT
        for i,val in enumerate(vision['DOWN_RIGHT']):
            place_value(val, x+(i+1),y+(i+1),i+1,True)
        
            
        return spatial      

    def __localRegion__(self):
        """Not used!"""
        # first lets get our full rgb image and pad it out 
        pix_square_size = (
            self.window_size//self.min_size
        )
        pixels = self._render_frame()
        pixels = torch.tensor(pixels)
        vision = self.vision_len
        vision_ext = int(vision*pix_square_size)
        padded_pixels = torch.nn.functional.pad(pixels.permute(2,0,1),
                                                pad=(vision_ext,vision_ext,vision_ext,vision_ext),
                                                mode='constant',value=0)
        local = []
        for a in range(self.num_agents):
            x = self.agent_positions[a]%self.n_cols
            y = self.agent_positions[a]//self.n_cols
            point = np.array([x,y],dtype=int) + vision
            
            center = pix_square_size * point + pix_square_size//2
            x = int(center[0])
            y = int(center[1])
            region = padded_pixels[ :, y-vision_ext-pix_square_size//2:y+vision_ext + pix_square_size//2, 
                                   x-vision_ext - pix_square_size//2:x+vision_ext + pix_square_size//2]
            local.append(region.permute(1,2,0))
        return local
          
    def __get_path__(self,pos,t_pos):
        x = pos % self.n_cols
        y = pos // self.n_cols
        x_t = t_pos % self.n_cols
        y_t = t_pos // self.n_cols
        # -- get path from their initial location to their target -- #
        agents_path = self.maze.find_shortest_path(c_start = (x,y),
                                                c_end = (x_t,y_t))
        return agents_path

    
    def _init_agents(self):
        """ Initialize the agents: this includes finding their initial positions,
         the initial target positions, and subsequent information like being done, 
          or the path """

        agent_positions = []
        agent_goals = []
        agents_done = []
        agents_path = []

        pos_set = set()
        # -- first lets put in our positions that are initialized by the user -- #
        if self.init_pos['agents'] !=None:
            pos_set = pos_set | set(self.init_pos['agents'])
            agent_positions = self.init_pos['agents'].copy()
        
        if self.init_pos['targets']!=None:
            pos_set = pos_set | set(self.init_pos['targets'])
            agent_goals = self.init_pos['targets'].copy()

        # -- go through each agent and initalize the the positions if they have not been initalized -- #
        for a in range(self.num_agents):

            # -- first lets get our agent positions -- #
            if self.init_pos['agents'] == None:
                pos = self.agent_target_obs_space.sample()
                while pos in pos_set:
                    pos = self.agent_target_obs_space.sample()
                pos_set.add(pos)
                agent_positions.append(pos)
            else:
                pos = self.init_pos['agents'][a]

            # -- second lets get our target positions for the agent
            if self.init_pos['targets'] == None:
                t_pos = self.agent_target_obs_space.sample()
                while t_pos in pos_set:
                    t_pos = self.agent_target_obs_space.sample()
                agent_goals.append(t_pos)
            else:
                t_pos = self.init_pos['targets'][a]

            # -- append False for the agent is done parameter -- #
            agents_done.append(False)

            # -- get minimal path to goal -- #
            agents_path.append(self.__get_path__(pos,t_pos))
                
        
        self.agent_positions = agent_positions
        self.agent_goals = agent_goals
        self.agents_done = agents_done
        self.agents_path = agents_path
        
    
    def manhattan_dist(self,point1, point2):
        x_1 = point1 % self.n_cols
        x_2 = point2 % self.n_cols
        y_1 = point1// self.n_cols
        y_2 = point2//self.n_cols
        return abs(x_1 - x_2) + abs(y_1 - y_2)
    
    def _get_vision(self,a):
        """ This looks in each of the directions of the agent and detects what it can see,
        up to some vision.
        0 = nothing
        1 = wall
        2 = anoter agent
        3 = personal goal"""

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
        agent_positions = self.agent_positions.copy()
        
        pos = agent_positions[a]
        
        # agents that are not done
        agents_not_done = np.where(np.array(self.agents_done) == False)[0]
        
        
        
        agent_pos_set = {agent_positions[i] for i in agents_not_done }
        
        agent_goal = self.agent_goals[a]
        
        # UP
        up_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = (pos_temp//n_cols<=0 or down_con_list[pos_temp - n_cols] == False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                up_vision += rest_wall
                break
            pos_temp -= n_cols
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    up_vision.append(2)
                elif  pos_temp == agent_goal:
                    up_vision.append(3)
                else:
                    up_vision.append(0)
            
        vision['UP'] = up_vision

        # DOWN 
        down_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = (pos_temp//n_cols>=n_rows-1 or down_con_list[pos_temp]==False )
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                down_vision += rest_wall
                break
            pos_temp += n_cols
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    down_vision.append(2)
                elif pos_temp == agent_goal:
                    down_vision.append(3)
                else:
                    down_vision.append(0)
            
        vision['DOWN'] = down_vision

        # LEFT 
        left_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = (pos_temp % n_cols <= 0 or right_con_list[pos_temp - 1]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                left_vision += rest_wall
                break
            pos_temp -= 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    left_vision.append(2)
                elif pos_temp ==  agent_goal:
                    left_vision.append(3)
                else:
                    left_vision.append(0)
            
        vision['LEFT'] = left_vision

        # RIGHT
        right_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = (pos_temp % n_cols >= n_cols-1 or right_con_list[pos_temp]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                right_vision += rest_wall
                break
            pos_temp +=1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    right_vision.append(2)
                elif pos_temp == agent_goal:
                    right_vision.append(3)
                else:
                    right_vision.append(0)
            
        vision['RIGHT'] = right_vision

        # UP LEFT
        up_left_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = ((pos_temp % n_cols <= 0 or pos_temp//n_cols<=0) or
                  (right_con_list[pos_temp-n_cols-1]==False or down_con_list[pos_temp-n_cols]==False)
                   and (down_con_list[pos_temp-n_cols-1]==False or right_con_list[pos_temp - 1]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                up_left_vision += rest_wall
                break
            pos_temp = pos_temp -1 - n_cols
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    up_left_vision.append(2)
                elif pos_temp == agent_goal:
                    up_left_vision.append(3)
                else:
                    up_left_vision.append(0)
            
        vision['UP_LEFT'] = up_left_vision

        # UP RIGHT
        up_right_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = ((pos_temp % n_cols >= n_cols-1 or pos_temp//n_cols<=0) or
                  (right_con_list[pos_temp-n_cols]==False or down_con_list[pos_temp-n_cols]==False)
                   and (down_con_list[pos_temp-n_cols+1]==False or right_con_list[pos_temp]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                up_right_vision += rest_wall
                break
            pos_temp = pos_temp - n_cols +1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    up_right_vision.append(2)
                elif pos_temp == agent_goal:
                    up_right_vision.append(3)
                else:
                    up_right_vision.append(0)
            
        vision['UP_RIGHT'] = up_right_vision

        # DOWN LEFT
        down_left_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = ((pos_temp % n_cols <= 0 or pos_temp//n_cols>=n_rows-1) or
                  (right_con_list[pos_temp-1]==False or down_con_list[pos_temp-1]==False)
                   and (down_con_list[pos_temp]==False or right_con_list[pos_temp+n_cols -1]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                down_left_vision += rest_wall
                break
            pos_temp  = pos_temp +n_cols -1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    down_left_vision.append(2)
                elif pos_temp == agent_goal:
                    down_left_vision.append(3)
                else:
                    down_left_vision.append(0)
            
        vision['DOWN_LEFT'] = down_left_vision

        # DOWN RIGHT
        down_right_vision = []
        pos_temp = pos
        for i in range(self.vision_len):
            
            v = ((pos_temp % n_cols >= n_cols-1 or pos_temp//n_cols>=n_rows-1) or
                  (right_con_list[pos_temp]==False or down_con_list[pos_temp+1]==False)
                   and (down_con_list[pos_temp]==False or right_con_list[pos_temp+n_cols]==False))
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                down_right_vision += rest_wall
                break
            pos_temp = pos_temp + n_cols + 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if pos_temp in agent_pos_set:
                    down_right_vision.append(2)
                elif pos_temp == agent_goal:
                    down_right_vision.append(3)
                else:
                    down_right_vision.append(0)
            
        vision['DOWN_RIGHT'] = down_right_vision

        return vision
        #return {
            #'UP': y == 0 or self.maze.connection_list[0][y-1][x] == False,
            #'DOWN': y == self.maze_shape[0]-1 or self.maze.connection_list[0][y][x] == False,
            #'LEFT': x == 0 or self.maze.connection_list[1][y][x-1] == False,
            #'RIGHT': x == self.maze_shape[1] - 1 or self.maze.connection_list[1][y][x] == False
        #}

    def reset(self,seed=None,options=None):
        if options!=None:
            self.new_maze(options['new_maze'])
            return self.reset()
        else:
            super().reset(seed=seed)
            self.timer = 0
            self._init_agents()
            state = self._get_state()
            info = self._get_info()
            
            if self.render_mode == 'human':
                self._render_frame()
            
            return state, info
        
    def _get_state(self):

        if self.obs_type == 'basic':
            state = {}
            for a in range(self.num_agents):
                state[f'agent_{a}'] = self.agent_positions[a]
                state[f'target_{a}'] = self.agent_goals[a]
            return state
        elif self.obs_type == 'rgb':
            # Note that this is in (window,window,3)
            return self._render_frame()
        elif self.obs_type == 'spatial':
            state = {}
            for a in range(self.num_agents):
                state[f'local_{a}'] = self.__spatial__(a)
                state[f'global_{a}'] = self.__global__(a)
                
            return state
 
    def _get_info(self):
        info = {}
        for a in range(self.num_agents):
            pos = self.agent_positions[a]
            t_pos = self.agent_goals[a]
            vision = self._get_vision(a)
            goal_dist = self.manhattan_dist(pos,t_pos)
            info[f'agent_{a}'] = {'UP_vision' : vision['UP'],
                                  'DOWN_vision': vision['DOWN'],
                                  'LEFT_vision': vision['LEFT'],
                                  'RIGHT_vision': vision['RIGHT'],
                                  'UP_LEFT_vision': vision['UP_LEFT'],
                                  'UP_RIGHT_vision': vision['UP_RIGHT'],
                                  'DOWN_LEFT_vision': vision['DOWN_LEFT'],
                                  'DOWN_RIGHT_vision': vision['DOWN_RIGHT'],
                                  'man_dist': goal_dist,
                                  'done' : self.agents_done[a],
                                  'path' : self.agents_path[a]}

            if self.obs_type == 'rgb' or self.obs_type=='spatial':
                info[f'agent_{a}']['pos'] = pos
                info[f'agent_{a}']['target'] = t_pos
        info['timer']=self.timer
        info['len_game'] = self.len_game
        info['n_agents'] = self.num_agents
        info['max_pos'] = self.n_cols*(self.n_rows) - 1
        
        return info
    
    def step(self,actions):
        
        # Furthermore, if pos is the position
        # going down: pos + n_cols
        # going up: pos - n_cols
        # going right: pos + 1
        # going left: pos - 1

        # --- get number of columns and rows --- #
        n_cols = self.n_cols

        agent_rewards = []

        # --- timer for how long the environment is going --- #
        self.timer+=1
        
        # --- go through each agent's action and process it --- #
        for i, action in enumerate(actions):

            rewards = 0
            pos = self.agent_positions[i]
            vision = self._get_vision(i)

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
            if self.agent_positions[i]==self.agent_goals[i] or self.agents_done[i]:
                
                self.agents_done[i] = True
                
            else:
                self.agent_positions[i] = pos
                self.agents_done[i] = False

            

            agent_rewards.append(rewards)

        # --- check if all agents are done --- #
        done = all(self.agents_done)


        # --- check if end of the time --- #
        truncated = self.timer == self.len_game

        if self.render_mode == 'human' and self.obs_type!='rgb':
            pixels = self._render_frame()
        
        return self._get_state(),np.array(agent_rewards), done, truncated, self._get_info()

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        
        
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

        self._draw_maze(screen,pix_square_size)

        # Draw the targets: 
        n_cols = self.n_cols
        n_rows = self.n_rows

        norm = Normalize(vmin=0,vmax=self.num_agents-1)
        colormap = plt.cm.get_cmap('tab10',self.num_agents)

        for i, point in enumerate(self.agent_goals):
            color = colormap(norm(i))
            x = point%n_cols
            y = point//n_cols
            pos = np.array([x,y])
            pygame.draw.rect(
                surface = screen, 
                color = tuple(int(c*255) for c in color[:3]), 
                rect=pygame.Rect(pix_square_size * pos, 
                            (pix_square_size,pix_square_size)
                            ),
                )
        self._draw_maze(screen,pix_square_size)
        # Draw the agents
        for i, point in enumerate(self.agent_positions):
            color = colormap(norm(i))
            x = point % n_cols
            y = point // n_cols
            pos = np.array([x,y])
            pygame.draw.circle(
                surface = screen,
                color = tuple(int(c*255) for c in color[:3]),
                center = (pos + 0.5) * pix_square_size,
                radius = pix_square_size/3,
            )

        


        if self.render_mode == 'human':
            self.window.blit(screen,screen.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)),axes = (1,0,2)
        )

    def _draw_maze(self,screen, square_size):
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
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
                    
                


