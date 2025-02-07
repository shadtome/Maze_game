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

class maze_env(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'],'render_fps':10,
                'obs_type': ['rgb','spatial','basic']}
    def __init__(self, maze, num_agents,vision_len = 1, render_mode = None, obs_type=None):
        super(maze_env, self).__init__()

        
        
        self.window_size = 512 #512

        self.maze = maze
        
        self.n_cols = maze.grid_shape[1]
        self.n_rows = maze.grid_shape[0]
        self.min_size = min(self.n_cols,self.n_rows)

        self.vision_len = vision_len
        self.num_agents = num_agents
        self.agent_positions = None
        self.agent_goals = None
        self.agents_done = None

        # Define observation and action spaces
        assert obs_type is None or obs_type in self.metadata['obs_type']
        self.obs_type = obs_type

        # Action space
        self.action_space = gym.spaces.Discrete(5)
        self.actions = ['STAY','UP','DOWN','LEFT','RIGHT']

        # Observation Space, depending on the obs_type
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
            None

        


        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode


        self.window = None
        self.clock = None

    def __basicSpatial__(self):
        grid_size = 2*self.vision_len + 1
        spatial_list = []
        for a in range(self.num_agents):
            vision = self._get_vision(a)
            spatial = torch.zeros(grid_size,grid_size,3,dtype=torch.int)
            x = self.vision_len
            y = self.vision_len
            spatial[y][x][0] = 0
            spatial[y][x][1] = 0
            spatial[y][x][2] = 255

            def place_value(val,x,y):
                if val == 0:
                    spatial[y][x][0] = 255
                    spatial[y][x][1] = 255
                    spatial[y][x][2] = 255
                elif val == 2:
                    spatial[y][x][0] = 255
                    spatial[y][x][1] = 0
                    spatial[y][x][2] = 0
                elif val == 3:
                    spatial[y][x][0] = 0
                    spatial[y][x][1] = 255
                    spatial[y][x][2] = 0
            
            # UP 
            for i,val in enumerate(vision['UP']):
                place_value(val,x,y-(i+1))
            
            # DOWN 
            for i,val in enumerate(vision['DOWN']):
                place_value(val,x,y+(i+1))

            # LEFT 
            for i,val in enumerate(vision['LEFT']):
                place_value(val,x - (i+1),y)

            # RIGHT
            for i,val in enumerate(vision['RIGHT']):
                place_value(val,x + (i+1),y)
            spatial_list.append(spatial)
        return spatial_list

        

    def __localRegion__(self):
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

        
    def _init_agents(self):
        
        agent_positions = []
        agent_goals = []
        agents_done = []
        pos = self.observation_space.sample()
        for a in range(self.num_agents):
            
            agent_positions.append(pos[f'agent_{a}'])
            agent_goals.append(pos[f'target_{a}'])
            agents_done.append(False)
        
        self.agent_positions = agent_positions
        self.agent_goals = agent_goals
        self.agents_done = agents_done
    
    def manhattan_dist(self,point1, point2):
        return abs(point1-point2)
    
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
        pos = pos.copy()
        
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

        return vision
        #return {
            #'UP': y == 0 or self.maze.connection_list[0][y-1][x] == False,
            #'DOWN': y == self.maze_shape[0]-1 or self.maze.connection_list[0][y][x] == False,
            #'LEFT': x == 0 or self.maze.connection_list[1][y][x-1] == False,
            #'RIGHT': x == self.maze_shape[1] - 1 or self.maze.connection_list[1][y][x] == False
        #}

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
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

    
    def _get_info(self):
        info = {}
        for a in range(self.num_agents):
            pos = self.agent_positions[a]
            vision = self._get_vision(a)
            goal_dist = self.manhattan_dist(pos,self.agent_goals[a])
            info[f'agent_{a}'] = {'UP_vision' : vision['UP'],
                                  'DOWN_vision': vision['DOWN'],
                                  'LEFT_vision': vision['LEFT'],
                                  'RIGHT_vision': vision['RIGHT'],
                                  'man_dist': goal_dist,
                                  'done' : self.agents_done[a]}
        #info['rgb'] = self._render_frame()
        info['local'] = self.__localRegion__()
        info['spatial'] = self.__basicSpatial__()
        return info
    
    def step(self,actions):

        # Furthermore, if pos is the position
        # going down: pos + n_cols
        # going up: pos - n_cols
        # going right: pos + 1
        # going left: pos - 1

        n_cols = self.n_cols
        n_rows = self.n_rows

        agent_rewards = []
        
        for i, action in enumerate(actions):
            rewards = 0
            pos = self.agent_positions[i]
            vision = self._get_vision(i)

            # move agent based on action, or don't move it if there is a wall
            if action == 1 and vision['UP'][0]!=1:
                pos -= n_cols
            elif action == 2 and vision['DOWN'][0]!=1:
                pos += n_cols
            elif action == 3 and vision['LEFT'][0]!=1:
                pos -= 1
            elif action == 4 and vision['RIGHT'][0]!=1:
                pos += 1
            else:
                if action !=0:
                    rewards -=0.01

            if self.agent_positions[i]==self.agent_goals[i] or self.agents_done[i]:
                rewards +=1
                self.agents_done[i] = True
                
            else:
                rewards -=0.01
                self.agent_positions[i] = pos
                self.agents_done[i] = False

            agent_rewards.append(rewards)

        done = all(self.agents_done)

        #if self.render_mode == 'human':
        if self.obs_type == 'basic' or self.obs_type == '':
            pixels = self._render_frame()
        
        return self._get_state(),np.array(agent_rewards), done, False, self._get_info()

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
        down_con = self.maze.connection_list[0]
        right_con = self.maze.connection_list[1]
        rows = self.n_rows
        cols = self.n_cols
        for row in range(rows):
            for col in range(cols):

                x = col * square_size
                y = row * square_size

                #pygame.draw.rect(screen,(255,255,255),(x,y,square_size,square_size),0)
                #pygame.draw.rect(screen, (0,0,0),(x,y,square_size,square_size),1)

                if col < cols - 1 and not right_con[row][col]:
                    pygame.draw.line(screen, (0,0,0), (x + square_size, y),
                                      (x + square_size, y + square_size),5)

                if row < rows - 1 and not down_con[row][col]:
                    pygame.draw.line(screen, color=(0,0,0), start_pos=(x, y+square_size),
                                     end_pos=(x+square_size,y+square_size),width=5)
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
                    
                


