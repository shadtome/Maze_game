
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import gymnasium as gym
import DQN.agents.basic as basic
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torch.optim.lr_scheduler import StepLR
import os
import device
import pandas as pd
import json
import seaborn as sns
import DQN.buffers as buffers
from DQN.schedulers.learning_rate.learning_rate_heads import VariableLR
from DQN.schedulers.epsilon_decay.basic import BaseEpsilonScheduler
from DQN.schedulers.epsilon_decay.epsilonLevels import GradientEpsilonScheduler
import Maze_env.wrappers.stickAction as sA
from DQN.training.basic import BaseTraining,MAZE_UPDATE,RANDOM_STATE



class CurriculumTraining(BaseTraining):
    def __init__(self,name,maze_dataset,maze_agent,
                 len_game = 1000,
                 n_agents=1,
                 replay_buffer_size = 100000,
                 replay_buffer_min_perc = 0.1,
                 policy_update = 4,
                 target_update = 1000,
                 gamma=0.99,tau = 0.01, batch_size=32, lambda_entropy = 0.01, 
                 lr=1e-3,lr_heads=1e-3, lr_step_size = 1000, lr_gamma = 0.1,lr_head_step_size = 1000,
                 lr_head_gamma = 0.1,l2_regular = 1e-4,
                 start_epsilon=1,final_epsilon=0.1,n_frames=50000,
                 beta = 0.4, alpha = 0.6, decay_total = 10000, per = False,
                 agent_pos = None, target_pos = None, 
                 curriculum_alpha = 1.0, curriculum_mu = 1.0):
        
        n_levels = maze_agent.n_heads



        super().__init__(name,maze_dataset,maze_agent,len_game,n_agents,
                         replay_buffer_size,replay_buffer_min_perc,policy_update,
                         target_update,gamma,tau,batch_size,lambda_entropy,lr,
                         lr_step_size,lr_gamma,l2_regular,start_epsilon,final_epsilon,
                         beta,alpha,decay_total,per,agent_pos,target_pos,
                         curriculum_mu=curriculum_mu,curriculum_alpha=curriculum_alpha,
                         n_levels = n_levels,lr_heads = lr_heads,
                         head_step_size = lr_head_step_size, head_gamma = lr_head_gamma)
        
        self.start_dist = self.get_start_dist()
        self.epsilon_upgrades = []
        self.distance_upgrades = []

    def __setup_epsilon_policy__(self, start_epsilon, end_epsilon, decay_total, n_levels,
                                 curriculum_mu,curriculum_alpha, **kwargs):
        return GradientEpsilonScheduler(start_epsilon,end_epsilon,decay_total,
                                        decayType='exponential',n_levels = n_levels,
                                        mu=curriculum_mu,alpha = curriculum_alpha) 

    def __setup_optimizer__(self, Q_fun, lr, l2_regular,lr_heads, **kwargs):

        param_groups = [
            {'params': Q_fun.base.parameters(), 'lr':lr},
            {'params': Q_fun.combine_base.parameters(), 'lr':lr}
        ]

        for heads in Q_fun.heads:
            param_groups.append({'params': heads.parameters(), 'lr':lr_heads})

        optimizer = torch.optim.Adam(param_groups,weight_decay=l2_regular)
        return optimizer

        
    def __setup_lr_scheduler__(self, optimizer, step_size, gamma,head_step_size,head_gamma, **kwargs):
        return VariableLR(optimizer,step_size,head_step_size,gamma,head_gamma)
        return super().__setup_lr_scheduler__( optimizer, step_size, gamma, **kwargs)

    def applyPolicyQ_fun(self, local_s, global_s, info):
        if isinstance(info,dict):
            head = torch.tensor(np.array([self.agents.get_head(info['dist'])]),dtype=int,device=device.DEVICE)
        else:
            head = torch.tensor(np.array([self.agents.get_head(d['dist']) for d in info]),dtype=int,device=device.DEVICE)
        return self.agents.Q_fun(local_s,global_s,head)
    
    def applyTargetQ_fun(self, local_s, global_s, info):
        if isinstance(info,dict):
            head = torch.tensor(np.array([self.agents.get_head(info['dist'])]),dtype=int,device=device.DEVICE)
        else:
            head = torch.tensor(np.array([self.agents.get_head(d['dist']) for d in info]),dtype=int,device=device.DEVICE)
        return self.target_Q_net(local_s,global_s,head)

    def __update_epsilon_scheduler__(self,frame):
        updated = self.epsilonScheduler.step()
                
        if updated['dist']:
            # -- increase the starting distance -- #
            self.start_dist = self.get_start_dist()
            print(f'Increasing Distance to {self.start_dist}')
            self.distance_upgrades.append(frame)
        if updated['level']:
            # -- remember that we upgraded the level -- #
            self.epsilon_upgrades.append(frame)

    def get_start_dist(self):
        return self.agents.cur_dist(self.epsilonScheduler.cur_level-1)
        
    
    def update_networks(self,update_start,frame):
        super().update_networks(update_start,frame)
        if update_start and frame % 10000 == 0:  
            print(f'Epsilon Level: {self.epsilonScheduler.cur_level}')
            print(f'Start dist: {self.start_dist}') 

    def setup_environment(self, maze, agent_pos, target_pos, **kwargs):
        return super().setup_environment(maze, agent_pos, target_pos,start_dist = self.start_dist, **kwargs)

    def reset_environment(self, env, **kwargs):
        return super().reset_environment(env,start_dist = self.start_dist, **kwargs)
                

    def get_action(self,env,state,info):

        # -- levels of the epsilon -- #
        epsilon = self.epsilonScheduler.epsilon
        actions = []
        # --- get action --- #
        for a in range(self.n_agents):
            dist_epsilon = 1.0
            cur_dist = info[f'agent_{a}']['dist']
            if self.start_dist == None:
                dist_epsilon = epsilon[-1]
            elif cur_dist<=self.start_dist:
                dist_epsilon = epsilon[self.agents.get_head(cur_dist)]  
            
            actions.append(self.agents.get_single_agent_action(
                    env = env,
                    state=state,
                    a=a,
                    info=info,
                    epsilon = dist_epsilon
                ))

        # --- save actions for results --- #
        self.actions_taken.append(actions)

        return actions
    
    def in_training_test(self, maze, **kwargs):
        return super().in_training_test(maze, start_dist = self.start_dist, **kwargs)
    
    def test_success_rate(self, frame, **kwargs):
        return super().test_success_rate(frame,start_dist = self.start_dist, **kwargs)


    def __additional_graphs__(self, axe):
        super().__additional_graphs__(axe)

        for frame in self.epsilon_upgrades:
            axe[0][0].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
            axe[1][0].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
            axe[1][1].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
            
        for frame in self.distance_upgrades:
            axe[0][0].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')
            axe[1][0].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')
            axe[1][1].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')

        for a in range(2,self.n_agents+2):
            
            for frame in self.epsilon_upgrades:
                axe[a][0].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
                axe[a][1].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
                
            for frame in self.distance_upgrades:
                axe[a][0].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')
                axe[a][1].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')


    def __getModelParam__(self):
        param = super().__getModelParam__()
        param['type_training'] = 'curriculum'
        param['curriculum_mu'] = self.epsilonScheduler.mu
        param['curriculum_alpha'] = self.epsilonScheduler.alpha

        return param
    
    

    