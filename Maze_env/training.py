import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import os

from Maze_env.env.mazes import maze_env
from agent import maze_agent

class Q_training:
    def __init__(self,start_epsilon = 1, final_epsilon=0.1,n_episodes = 100,update_factor = 500):

        self.agent = maze_agent([0,0],[1,1],)