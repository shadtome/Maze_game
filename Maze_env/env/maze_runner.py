from Maze_env.env.mazes import BasicMaze
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


class MazeRunner(BasicMaze):
    def __init__(self, maze, len_game=1000, 
                 num_objects={'agents':1}, 
                 vision_len={'agents':1}, 
                 type_of_objects=['agents'], 
                 objectives={'agents': 'goal'}, 
                 action_type='full', 
                 render_mode=None, 
                 obs_type=None, 
                 init_pos=None, 
                 start_dist=None, 
                 dist_paradigm='radius', 
                 collision_rules=None,
                 colormap = {'agents':'tab10'}):
        
        super().__init__(maze, len_game, num_objects, vision_len, 
                         type_of_objects, objectives, action_type, 
                         render_mode, obs_type, init_pos, start_dist, 
                         dist_paradigm, collision_rules,colormap)
        
    def __place_value__(self, val, alpha, beta, diag=False):
        if val == self.vision_mapping['goal'] and diag == False:
                return [0,int(255* beta),0]
        elif val == self.vision_mapping['goal']:
            return [0,int(255* beta * alpha),0]
        return super().__place_value__(val, alpha, beta, diag)
    
    def __place_center_value__(self, val):
        if val == self.vision_mapping['goal']:
            return [0,255,0]
        return super().__place_center_value__(val)
        
    def __what_they_see__(self, pos, pos_set, type_object, a):
        if self.objectives[type_object] == 'goal':
            if pos == self.pos[type_object + '_goals'][a]:
                return self.vision_mapping['goal']
        return super().__what_they_see__(pos, pos_set, type_object, a)
        
    def __init_object_goal__(self,type_object,pos_set):
        if self.objectives[type_object] == 'goal':
            object_goals = []
            object_path = []

            if type_object+ '_goals' in self.init_pos:
                pos_set = pos_set | set(self.init_pos[type_object + '_goals'])

            for a in range(self.num_objects[type_object]):
                t_pos = self.__get_unique_pos__(type_object + '_goals',pos_set,a)
                object_goals.append(t_pos)

                object_path.append(self.__get_path__(self.pos[type_object][a],t_pos))

            self.pos[type_object+'_goals'] = object_goals
            self.path[type_object] = object_path


    def __get_object_info__(self, type_object):
        info = super().__get_object_info__(type_object)
        if type_object + '_goals' in self.pos:
            for a in range(self.num_objects[type_object]):
                pos = self.pos[type_object][a]
                t_pos = self.pos[type_object + '_goals'][a]
                goal_dist = self.get_dist(pos,t_pos)
                info[type_object+f'_{a}']['dist'] = goal_dist
                info[type_object + f'_{a}']['path'] = self.path[type_object][a]
                info[type_object + f'_{a}']['goal'] = t_pos
            info[type_object]['goal']=True
        else:
            info[type_object]['goal'] = False
        return info


    def __init_individual_objects__(self, type_object, pos_set):
        # -- do base initializing of objects -- #
        super().__init_individual_objects__(type_object, pos_set)
        # -- do the goal initaliziation for maze running -- #
        self.__init_object_goal__(type_object,pos_set)


    def __add_goal_to_env__(self,type_object,screen,pix_square_size):
        norm = Normalize(vmin=0,vmax=self.num_objects[type_object]-1)
        colormap = plt.cm.get_cmap(self.colormap[type_object],self.num_objects[type_object])
        if type_object+'_goals' in self.pos:
            for i, point in enumerate(self.pos[type_object+'_goals']):
                if self.done[type_object][i]==False:
                    
                    color = self.__get_color__(colormap,norm,i,type_object)
                    x,y = self.__getCoords__(point)
                    pos = np.array([x,y])
                    pygame.draw.rect(
                        surface = screen, 
                        color = tuple(int(c*255) for c in color[:3]), 
                        rect=pygame.Rect(pix_square_size * pos, 
                                    (pix_square_size,pix_square_size)
                                    ),
                        )
            if self.dist_paradigm == 'path':
                for i, path in enumerate(self.path[type_object]):
                    if self.done[type_object][i]==False:
                        color = self.__get_color__(colormap,norm,i,type_object)
                        for point in path[1:]:
                            x,y = self.__getCoords__(point)
                            pos = np.array([x,y])
                            pygame.draw.circle(
                                surface = screen,
                                color = tuple(int(c*255) for c in color[:3]),
                                #color = (255,0,0),
                                center = (pos + 0.5) * pix_square_size,
                                radius = pix_square_size/7,)
                            

    def __add_to_env__(self, type_object, screen, pix_square_size):
        self.__draw_maze__(screen,pix_square_size)
        self.__add_goal_to_env__(type_object,screen,pix_square_size)
        super().__add_to_env__(type_object, screen, pix_square_size)

    def __is_done__(self, type_object, pos, index):
        if self.objectives[type_object]=='goal':
            if self.pos[type_object][index]==self.pos[type_object + '_goals'][index]:
                    self.done[type_object][index] = True 
                    self.success[type_object][index]=True 
            else:
                # -- if the object has a end goal, find its minimal path -- #
                self.path[type_object][index] = self.__get_path__(pos,self.pos[type_object + '_goals'][index])
        return super().__is_done__(type_object, pos, index)