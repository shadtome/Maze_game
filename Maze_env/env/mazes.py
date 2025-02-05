import numpy as np
import matplotlib.pyplot as plt
import maze_dataset.maze.lattice_maze as lm
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot
import random
import gymnasium as gym
from matplotlib.colors import Normalize

import Maze_env.agent as agent
import pygame

# We will set up our state space as ((x,y))
# This is just the position for now

# 0 stay
# 1 move up
# 2 move down
# 3 move left
# 4 move right

class maze_env(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'],'render_fps':4}
    def __init__(self, maze, num_agents,vision_len = 1, render_mode = None):
        super(maze_env, self).__init__()
        
        self.window_size = 512

        self.maze = maze
        self.maze_shape = maze.grid_shape
        self.min_size = min(self.maze_shape)

        self.vision_len = vision_len
        self.num_agents = num_agents
        self.agent_positions = None
        self.agent_goals = None

        # Define observation and action spaces
        self.action_space = gym.spaces.Discrete(5)
        obs = dict()
        for a in range(self.num_agents):
            obs[f'agent_{a}'] = gym.spaces.Box(low=np.array([0,0]),
                                                high=np.array([self.maze_shape[1]-1,self.maze_shape[0]-1]),dtype = np.int64)
            obs[f'target_{a}']= gym.spaces.Box(low=np.array([0,0]),
                                                high=np.array([self.maze_shape[1]-1,self.maze_shape[0]-1]),dtype = np.int64)
        self.observation_space = gym.spaces.Dict(obs)

        self.actions = ['STAY','UP','DOWN','LEFT','RIGHT']


        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        
    def _init_agents(self):
        
        agent_positions = []
        agent_goals = []
        pos = self.observation_space.sample()
        for a in range(self.num_agents):
            agent_positions.append(pos[f'agent_{a}'])
            agent_goals.append(pos[f'target_{a}'])
        self.agent_positions = agent_positions
        self.agent_goals = agent_goals
    
    def manhattan_dist(self,point1, point2):
        return abs(point1[0] - point2[0] ) + abs(point1[1] - point2[1])
    
    def _get_vision(self,a):
        """ This looks in each of the directions of the agent and detects what it can see,
        up to some vision.
        0 = nothing
        1 = wall
        2 = anoter agent
        3 = personal goal"""

        vision = {}
        agent_positions = self.agent_positions.copy()
        pos = agent_positions.pop(a)
        x = pos[0]
        y = pos[1]
        agent_pos_set = {tuple(pos) for pos in agent_positions}
        agent_goal = tuple(self.agent_goals[a])

        # UP
        up_vision = []
        y_temp = y
        x_temp = x
        for i in range(self.vision_len):
            v = (y_temp <= 0 or self.maze.connection_list[0][y_temp-1][x_temp]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                up_vision += rest_wall
                break
            y_temp -= 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if (x_temp,y_temp) in agent_pos_set:
                    up_vision.append(2)
                elif np.array_equal(np.array([x_temp, y_temp]), agent_goal):
                    up_vision.append(3)
                else:
                    up_vision.append(0)
            
        vision['UP'] = up_vision

        # DOWN 
        down_vision = []
        y_temp = y
        x_temp = x
        for i in range(self.vision_len):
            v = (y_temp >= self.maze_shape[0]-1 or self.maze.connection_list[0][y_temp][x_temp]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                down_vision += rest_wall
                break
            y_temp += 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if (x_temp,y_temp) in agent_pos_set:
                    down_vision.append(2)
                elif np.array_equal(np.array([x_temp, y_temp]), agent_goal):
                    down_vision.append(3)
                else:
                    down_vision.append(0)
            
        vision['DOWN'] = down_vision

        # LEFT 
        left_vision = []
        y_temp = y
        x_temp = x
        for i in range(self.vision_len):
            v = (x_temp <= 0 or self.maze.connection_list[1][y_temp][x_temp-1]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                left_vision += rest_wall
                break
            x_temp -= 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if (x_temp,y_temp) in agent_pos_set:
                    left_vision.append(2)
                elif np.array_equal([x_temp, y_temp], agent_goal):
                    left_vision.append(3)
                else:
                    left_vision.append(0)
            
        vision['LEFT'] = left_vision

        # RIGHT
        right_vision = []
        y_temp = y
        x_temp = x
        for i in range(self.vision_len):
            v = (x_temp >= self.maze_shape[1] - 1 or self.maze.connection_list[1][y_temp][x_temp]==False)
            if v == True:
                rest_wall = [1 for _ in range(self.vision_len - i)]
                right_vision += rest_wall
                break
            x_temp += 1
            if v == False: # This means that there is not a wall here, so either this is another agent or goal
                if (x_temp,y_temp) in agent_pos_set:
                    right_vision.append(2)
                elif np.array_equal(np.array([x_temp, y_temp]), agent_goal):
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
        state = {}
        for a in range(self.num_agents):
            
            state[f'agent_{a}'] = self.agent_positions[a]
            state[f'target_{a}'] = self.agent_goals[a]

        return state
    
    def _get_info(self):
        info = {}
        for a in range(self.num_agents):
            x,y = self.agent_positions[a]
            vision = self._get_vision(a)
            goal_dist = self.manhattan_dist([x,y],self.agent_goals[a])
            info[f'agent_{a}'] = {'UP_vision' : vision['UP'],
                                  'DOWN_vision': vision['DOWN'],
                                  'LEFT_vision': vision['LEFT'],
                                  'RIGHT_vision': vision['RIGHT'],
                                  'man_dist': goal_dist}
        return info
    
    def step(self,actions):

        agent_rewards = []
        done_flags = []

        for i, action in enumerate(actions):
            rewards = 0
            x,y = self.agent_positions[i]
            vision = self._get_vision(i)

            # move agent based on action, or don't move it if there is a wall
            if action == 1 and vision['UP'][0]!=1:
                y -=1
            elif action == 2 and vision['DOWN'][0]!=1:
                y +=1
            elif action == 3 and vision['LEFT'][0]!=1:
                x -= 1
            elif action == 4 and vision['RIGHT'][0]!=1:
                x += 1
            else:
                if action !=0:
                    rewards -=0.01

            self.agent_positions[i] = np.array([x,y])
            
            if np.array_equal(self.agent_positions[i],self.agent_goals[i]):
                rewards +=1
                done_flags.append(True)
            else:
                rewards -=0.01
                done_flags.append(False)

            agent_rewards.append(rewards)

        done = all(done_flags)

        if self.render_mode == 'human':
            self._render_frame()

        return self._get_state(),np.array(agent_rewards), done, False, self._get_info()
    
    def render_pos(self):
        fig, ax = plt.subplots(figsize=(7,7))
        MazePlot(self.maze).plot(fig_ax=[fig,ax])
        colors = plt.cm.Set1(np.linspace(0,1,self.num_agents))
        goal_colors = plt.cm.inferno(np.linspace(0,1,self.num_agents))

        for i, point in enumerate(self.agent_positions):
            plt.scatter(7*(2*point[0]+1),7*(2*point[1]+1),color=colors[i],label=f'Agent {i}')
            
        for i, point in enumerate(self.agent_goals):
            plt.scatter(7*(2*point[0]+1),7*(2*point[1]+1),color=goal_colors[i],label=f'Goal {i}')

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12, title="Agents")
        plt.show()

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
        norm = Normalize(vmin=0,vmax=self.num_agents-1)
        colormap = plt.cm.get_cmap('tab10',self.num_agents)

        for i, point in enumerate(self.agent_goals):
            color = colormap(norm(i))
            pygame.draw.rect(
                surface = screen, 
                color = tuple(int(c*255) for c in color[:3]), 
                rect=pygame.Rect(pix_square_size * point, 
                            (pix_square_size,pix_square_size)
                            ),
                )
        self._draw_maze(screen,pix_square_size)
        # Draw the agents
        for i, point in enumerate(self.agent_positions):
            color = colormap(norm(i))
            pygame.draw.circle(
                surface = screen,
                color = tuple(int(c*255) for c in color[:3]),
                center = (point + 0.5) * pix_square_size,
                radius = pix_square_size/3,
            )

        


        if self.render_mode == 'human':
            self.window.blit(screen,screen.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels2d(screen)),axes = (1,0,2)
            )



    def _draw_maze(self,screen, square_size):
        down_con = self.maze.connection_list[0]
        right_con = self.maze.connection_list[1]
        rows = self.maze_shape[0]
        cols = self.maze_shape[1]
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
                    
                


