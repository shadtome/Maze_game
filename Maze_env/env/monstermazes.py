from Maze_env.env.maze_runner import MazeRunner
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

# IN PROGRESS:
# Need to finish this while adding in appropriate methods to deal with 
# the monster killing other agents

# Need to change the _get_vision__ in the original to deal with seeing agents
# from different object types;  like it can see runners it wants to eat, but ignores
# other monsters for example, or even makes them a different color
# so it learns to ignore them
# Add in the appripriate info to be used in the rewards function
# Need to define the rewards function and what would push the monster 
# to want to go towards its prey.  
class MonsterMaze(MazeRunner):
    def __init__(self, maze, len_game=1000, 
                 num_objects=..., vision_len=..., 
                 type_of_objects=..., objectives=..., 
                 action_type='full', render_mode=None, 
                 obs_type=None, init_pos=None, start_dist=None, 
                 dist_paradigm='radius', collision_rules=None, 
                 colormap=...,
                 target = {'monsters':'agents'}):
        super().__init__(maze, len_game, num_objects, vision_len, type_of_objects, 
                         objectives, action_type, render_mode, 
                         obs_type, init_pos, start_dist, dist_paradigm, 
                         collision_rules, colormap)
        self.target = target
        self.kill_targets = {}
        self.kill_target_paths = {}
        self.vision_mapping['kill_target'] = 4


    def __place_value__(self, val, alpha, beta, diag=False):
        if val == self.vision_mapping['kill_target'] and diag == False:
            return [int(160*beta),int(32*beta),int(240*beta)]
        elif val == self.vision_mapping['kill_target']:
            return [int(160*beta*alpha),int(32*beta*alpha),int(240*beta*alpha)]
        return super().__place_value__(val, alpha, beta, diag)
    
    def __place_center_value__(self, val):
        if val == self.vision_mapping['kill_target']:
            return [int(160),int(32),int(240)]
        return super().__place_center_value__(val)

    def __what_they_see__(self, pos, pos_set, type_object, a):
        if self.objectives[type_object] == 'kill':
            if pos == self.pos[self.target[type_object]][self.kill_targets[type_object][a]]:
                return self.vision_mapping['kill_target']
        return super().__what_they_see__(pos, pos_set, type_object, a)
        

    def __init_object_kill__(self,type_object):
        if self.objectives[type_object] == 'kill':
            object_kill_target = []
            object_kill_path = []
            for a in range(self.num_objects[type_object]):
                kill_target = np.random.randint(0,self.num_objects[self.target[type_object]])
                object_kill_target.append(kill_target)
                pos = self.pos[type_object][a]
                kill_pos = self.pos[self.target[type_object]][kill_target]
                object_kill_path.append(self.__get_path__(pos,kill_pos))

            self.kill_targets[type_object] = object_kill_target
            self.kill_target_paths[type_object] = object_kill_path


    def __get_object_info__(self, type_object):
        info = super().__get_object_info__(type_object)
        if self.objectives[type_object]=='kill':
            for a in range(self.num_objects[type_object]):
                kill_target = self.kill_targets[type_object][a]
                pos = self.pos[type_object][a]
                target_kill_pos = self.pos[self.target[type_object]][kill_target]
                goal_dist = self.get_dist(pos,target_kill_pos)
                info[type_object + f'_{a}']['kill_target'] = (self.target[type_object],kill_target)
                info[type_object + f'_{a}']['kill_dist'] = goal_dist
                info[type_object + f'_{a}']['kill_path'] = self.kill_target_paths[type_object][a]
                info[type_object + f'_{a}']['kill_target_pos'] = target_kill_pos
            info[type_object]['kill']=True
        else:
            info[type_object]['kill'] = False
        return info
    
    def __init_individual_objects__(self, type_object, pos_set):
        # -- do base initializing of objects -- #
        super().__init_individual_objects__(type_object, pos_set)
        # -- do the kill initaliziation for maze running -- #
        self.__init_object_kill__(type_object)


    def __add_kill_path_to_env__(self,type_object,screen,pix_square_size):
        norm = Normalize(vmin=0,vmax=self.num_objects[type_object]-1)
        colormap = plt.cm.get_cmap(self.colormap[type_object],self.num_objects[type_object])
        if self.objectives[type_object] == 'kill':
            if self.dist_paradigm == 'path':
                for i, path in enumerate(self.kill_target_paths[type_object]):
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
        self.__add_kill_path_to_env__(type_object,screen,pix_square_size)
        super().__add_to_env__(type_object, screen, pix_square_size)

    def __is_done__(self, type_object, pos, index):
        if self.objectives[type_object]=='kill':
            target_type = self.target[type_object]
            target_index = self.kill_targets[type_object][index]
            dead = self.dead[target_type]
            done = self.done[target_type]
            if self.dead[target_type][target_index] or self.done[target_type][target_index]:
                # -- pick next target -- #
                if all([a or b for a,b in zip(dead,done)]):
                    self.done[type_object][index] = True
                else:
                    new_target = np.random.randint(0,self.num_objects[target_type])
                    while done[new_target] or dead[new_target]:
                        new_target = np.random.randint(0,self.num_objects[target_type])
                    self.kill_targets[type_object][index] = new_target
                    pos = self.pos[type_object][index]
                    kill_pos = self.pos[target_type][new_target]
                    self.kill_target_paths[type_object][index] = self.__get_path__(pos,kill_pos)

                


        return super().__is_done__(type_object, pos, index)

            
        

    
        