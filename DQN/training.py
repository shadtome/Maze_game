import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import gymnasium as gym
import DQN.agent as agent
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
import DQN.schedulers as schedulers
import Maze_env.wrappers.stickAction as sA



#################
# ----- Fixed Hyperparameters ---------- #
MAZE_UPDATE = 100
RANDOM_STATE = 49



class Maze_Training:
    def __init__(self,name,maze_dataset,maze_agent,
                 len_game = 1000,
                 n_agents=1,
                 replay_buffer_size = 100000,
                 replay_buffer_min_perc = 0.1,
                 policy_update = 4,
                 target_update = 1000,
                 gamma=0.99,tau = 0.01, batch_size=32, lambda_entropy = 0.01, 
                 lr=1e-3, lr_step_size = 1000, lr_gamma = 0.1, l2_regular = 1e-4,
                 start_epsilon=1,final_epsilon=0.1,n_frames=50000,
                 beta = 0.4, alpha = 0.6, decay_total = 10000, per = False,
                 agent_pos = None, target_pos = None,
                 curriculum = False, threshold = 0.70, 
                 curriculum_alpha = 1.0, curriculum_mu = 1.0):
        """Used to train a deep Q-network for agents exploring a maze.
        name: string used for saving the name and for loading

        maze_dataset: a dataset of mazes that come from the maze_dataset package

        maze_agent: 
        n_agents: the maze enviroment can run multiple agents and you can accumulate their 
        experiences together.  But it is aimed to be adversarial in the sense we want to avoid
        others

        start_epsilon: this is the start epsilon for the greedy policy that is enacted and decreased
        as the agent gets better.

        final_epsilon: this is the final epsilon that it will decay to.

        n_episodes: number of iterations of the enviroment the agent will go through while training

        update_factor: This how many times the target Will change this"""


        # --- Dynamic Hyperparameters --- #
        self.gamma = gamma  # discount factor
        self.tau = tau   # soft update factor
        self.batch_size = batch_size  # batch size
        self.lambda_entropy = lambda_entropy # entropy regularization term
        self.l2_regular = l2_regular # l2 regularization term
        self.alpha = alpha # priority experience replay buffer priority 
        self.beta = beta # importance sampling correction
        self.replay_buffer_size = replay_buffer_size  # replay buffer size
        self.replay_buffer_min_perc = replay_buffer_min_perc
        self.policy_update = policy_update  # frequency of policy update (backpropogation)
        self.target_update = target_update  # frequency of target update (soft update)
        self.decay_total = decay_total # How long to do the epsilon decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        # The agent with the Q_function
        self.agents = maze_agent

        # The target Q_net
        self.target_Q_net = self.agents.Q_fun.copy()
        self.target_Q_net = self.target_Q_net.to(device=device.DEVICE)
        self.target_Q_net.eval()

        # Collection of mazes (can be single maze or multiple)
        self.mazes = maze_dataset

        # -- number of total agents -- #
        self.n_agents = n_agents

        # -- agent and target initial positions, if fixed -- #
        self.agent_pos = agent_pos
        self.target_pos = target_pos


        # -- maximum length of episodes -- #
        self.len_game = len_game

        # -- Replay Buffer -- #
        self.per = per  # signifies to use priority replay buffer
        
        if self.per:
            # priority buffer
            self.replay_buffer = buffers.PERBuffer(capacity=self.replay_buffer_size,alpha=alpha,beta=self.beta)
        else:
            # uniform buffer
            self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        # file name for saving and loading
        self.name = name


        # -- where to stop the epsilon decay policy with respect to number of total frames -- #
        self.decay_stop = int(n_frames*1)

        # --- epsilon-greedy policy --- #
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon

        # --- epsilon-greedy policy scheduler --- #
        self.curriculum = curriculum
        max_dist = self.mazes.shape[0] + self.mazes.shape[1] - 2
        if curriculum:
            
            self.epsilonScheduler = schedulers.epsilonDecayScheduler(start_epsilon=start_epsilon,
                                                                 end_epsilon=final_epsilon,
                                                                 decay_total=decay_total,
                                                                 threshold=threshold,
                                                                 n_levels=max_dist,
                                                                 alpha = curriculum_alpha,
                                                                 mu = curriculum_mu)
            self.start_dist = 1                                             
        else:
            self.epsilonScheduler = schedulers.epsilonDecayScheduler(start_epsilon=start_epsilon,
                                                                 end_epsilon=final_epsilon,
                                                                 decay_total=decay_total,
                                                                 threshold=threshold,
                                                                 n_levels=1)
            self.start_dist = max_dist

        # -- number of total frames -- #
        self.n_frames = self.epsilonScheduler.total_time() + int(self.replay_buffer_size*self.replay_buffer_min_perc)

        # -- learning rate -- #
        self.lr = lr

        # -- loss function -- #
        #self.loss_fun = nn.MSELoss()
        self.loss_fun = nn.SmoothL1Loss(reduction='none')  # The Huber loss function, more stable to outliers

        # Optimizer
        #self.optimizer = torch.optim.SGD(self.agents.Q_fun.parameters(),lr=self.lr)
        self.optimizer = torch.optim.Adam(self.agents.Q_fun.parameters(),lr=self.lr,amsgrad=False,weight_decay = l2_regular)
        #self.optimizer = torch.optim.RMSprop(self.agents.Q_fun.parameters(),lr=lr)

        # -- scheduler for the learning rate -- #
        self.scheduler = StepLR(self.optimizer,
                                step_size=lr_step_size,
                                gamma=lr_gamma)
        

        # -- lists for saving the results -- #
        self.losses = []
        self.scores = []
        self.cum_reward = {}
        self.actions_taken = []
        self.Q_values = {}
        self.td_errors = {}
        for a in range(self.agents.n_actions):
            self.Q_values[a] = []
            

        for a in range(self.n_agents):
            self.cum_reward[f'agent_{a}'] = []
            self.td_errors[a] = []

        # -- save the best agent -- #
        self.best_agent = self.agents
        self.best_score = 0.0

        self.epsilon_upgrades = []
        self.distance_upgrades = []
        


    def soft_update(self,tau):
        """ Used to update the current target Q-network by the policy Q-network by a weighted sum:
            i.e., target = tau*policy + (1-tau)*target.  This ensures that the training is smoother:
            
            tau: this is a number in [0,1], where tau=1 implies that we replace the target parameters
            with the policy parameters.  We use a small tau to make the policy slowly update the target."""
        
        # --- calculate target = tau (policy) + (1-tau) target --- #
        for target_param, policy_param in zip(self.target_Q_net.parameters(),self.agents.Q_fun.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def transform(self,local_s,global_s,action,next_local_s,next_global_s,reward,terminated,weight=None,single=False):

        # we need everything to be a tensor to put into our neural network,
        # furthermore, we will need to permute our shape, since gymnasium outputs images
        # as (h,w,c)
        local_st = torch.tensor(np.array(local_s),dtype=torch.float32,device = device.DEVICE)
        if single:
            local_st = local_st.unsqueeze(0)
        local_st = local_st.permute((0,3,1,2))
        local_st = local_st/255

        global_st = torch.tensor(np.array(global_s),dtype=torch.float32,device = device.DEVICE)
        
        if single:
            global_st = global_st.unsqueeze(0)
        
        global_st[:,0:2] = global_st[:,0:2]/(self.agents.CNN_shape[1]*self.agents.CNN_shape[2]-1)
        global_st[:,3] = global_st[:,3]/(self.agents.CNN_shape[1] + self.agents.CNN_shape[2])

        if single:
            action_t = torch.tensor(np.array([action]), dtype=torch.int64,device = device.DEVICE)
        else:
            action_t = torch.tensor(np.array(action), dtype=torch.int64,device = device.DEVICE)

        next_local_st = torch.tensor(np.array(next_local_s),dtype=torch.float32,device=device.DEVICE)
        if single:
            next_local_st = next_local_st.unsqueeze(0)
        next_local_st = next_local_st.permute((0,3,1,2))

        next_global_st = torch.tensor(np.array(next_global_s),dtype=torch.float32,device = device.DEVICE)
        if single:
            next_global_st = next_global_st.unsqueeze(0)


        
        if single:
            reward_t = torch.tensor(np.array([reward]),dtype=torch.float32,device = device.DEVICE)
            reward_t = reward_t.unsqueeze(0)
            terminated_t = torch.tensor(np.array([terminated]),dtype = torch.int64,device= device.DEVICE)
            terminated_t = terminated_t.unsqueeze(0)
        else:
            reward_t = torch.tensor(np.array(reward),dtype=torch.float32,device = device.DEVICE)
            terminated_t = torch.tensor(np.array(terminated),dtype = torch.int64,device= device.DEVICE)
        
        if weight is not None:
            if single:
                weight_t = torch.tensor(np.array([weight]),dtype=torch.float32,device=device.DEVICE)
            else:
                weight_t = torch.tensor(np.array(weight),dtype = torch.float32,device = device.DEVICE )
            return local_st,global_st,action_t,next_local_st,next_global_st,reward_t, terminated_t, weight_t

        return local_st,global_st,action_t,next_local_st,next_global_st,reward_t, terminated_t


    def sample_replay(self, batch_size):
        """ Used to sample a collection of experiences fro the replay buffer that the agents have done:
        i.e., it is filled with initial state, action take, then the resulting state with corresponding reward"""

        # sample from the buffer
        if self.per:
            batch, weights = self.replay_buffer.sample(batch_size)
        
            sample = [exp for _, exp ,_  in batch]
        else:
            random.seed(RANDOM_STATE)
            sample = random.sample(self.replay_buffer,batch_size)
        
        # get the information

        local_state,global_state, action, next_local_state, next_global_state, reward,terminated= zip(*sample)
        
        return self.transform(local_state,global_state, action, next_local_state, next_global_state, reward,terminated,weights)

    def compute_loss(self,frame):
        """This is to compute the loss between the target Q-network and the policy Q-network.
        This also adds a entropy term.  The entropy terms wants to be maximized so that it is not
        determinisitic, i.e., staying in one spot forever.
        
        gamma: a regularizing term that ensures convergence a value in (0,1)
        lambda_entropy: this is the regularization term for the entropy loss, here 
                        we want to maximize entropy."""
        
        # --- get a sample of replays of past experiences --- #
        local_s,global_s, actions,next_local_s,next_global_s, rewards, terminated, weight = self.sample_replay(self.batch_size)
        
        # We need to find $Q^*(s,a) \approx r + \gamma * Q(s', max_{a'} Q'(s',a'))
        # Where (s,a,r,s') is from the replay buffer, Q is the policy net, Q' is the target net


        # --- get Q(s,a) for each action --- #
        q_values = self.agents.Q_fun(local_s,global_s)

        avg_q_values = q_values.mean(dim=0).tolist()
        for a in range(self.agents.n_actions):
            self.Q_values[a].append(avg_q_values[a])
        
        # --- pick the Q values corresponding to the picked actions --- #
        selected_q_values = q_values.gather(1,actions.unsqueeze(1)).squeeze()
        
        """
        with torch.no_grad():
    next_q_values = q_net(next_state)  # Use the online network
    best_actions = torch.argmax(next_q_values, dim=1)  # Select best actions
    target_q_values = target_net(next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)  # Use target network values
expected_q = reward + gamma * target_q_values * (1 - done)
"""

        with torch.no_grad():
            # --- get next actions where Q(s',a) is maximized --- #
            next_actions = self.agents.Q_fun(next_local_s, next_global_s).argmax(1, keepdim=True)

            # --- calculate Q'(s',a') where a' is the actions maximies from Q(s',a) above --- #
            next_q_values = self.target_Q_net(next_local_s,next_global_s).detach().gather(1,next_actions).squeeze(1)
            
            # --- calculate the estimator of the Bellmann equation --- #
            target = rewards + self.gamma*next_q_values * (1-terminated.float())

        # --- compute the loss between Q(s,a) and the Bellmann equation --- #
        loss = self.loss_fun(selected_q_values,target)
        loss = (weight * loss).mean()
        
        
        # --- Calculate the probabilities p(a|s) to compute entropy--- #
        action_probs = torch.softmax(q_values,dim=1)

        # --- compute the entropy of the distribution p(-|s)  --- #
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-6),dim=1)

        #if frame% 10000 == 0:
        #    print(f"Max Q-value: {selected_q_values.max().item()}, Max Target Values {target.max().item()}, minus: {(target- selected_q_values).max().item()}")
        #    print(f'Max Entropy: {entropy.max().item()}, Min Entropy: {entropy.min().item()}')
        
        # --- we want to maximize entropy to make it less deterministic, so we take the negative --- #
        loss -=self.lambda_entropy * entropy.mean()
        
        return loss, action_probs
    
    def td_error(self,local_s,global_s,action,n_local_s,n_global_s,reward,terminated):
        """Compute the td-error:
            r + gamma*Q'(s',argmax_{a'}Q(s'a')) - Q(s,a)"""
        local_s,global_s,action,n_local_s,n_global_s,reward,terminated = self.transform(local_s,global_s,action,n_local_s,n_global_s,reward,terminated,single=True)

        # --- get Q(s,a) for each action --- #
        q_values = self.agents.Q_fun(local_s,global_s)


        # --- pick the Q values corresponding to the picked actions --- #
        selected_q_values = q_values.gather(1,action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # --- get next actions where Q(s',a) is maximized --- #
            next_actions = self.agents.Q_fun(n_local_s, n_global_s).argmax(1, keepdim=True).detach()

            # --- calculate Q'(s',a') where a' is the actions maximies from Q(s',a) above --- #
            next_q_values = self.target_Q_net(n_local_s,n_global_s).gather(1,next_actions).squeeze(1)
            
            # --- calculate the estimator of the Bellmann equation --- #
            target = reward + self.gamma*next_q_values * (1-terminated.float())

        td_e = target - selected_q_values
        
        return td_e.detach().cpu().numpy()
    
    def append_to_RB(self,local_s,global_s,action,n_local_s,n_global_s,reward,terminated,agent_id):
        """Append the experience to the replay buffer depending on the type of buffer"""

        td_e = self.td_error(local_s,global_s,action,n_local_s,n_global_s,reward,terminated)
        self.td_errors[agent_id].append(td_e)

        if self.per:
            self.replay_buffer.add((local_s,global_s,action,
                                    n_local_s,n_global_s,
                                    reward,terminated),td_e)
        else:
            self.replay_buffer.append([local_s,global_s,action,
                                            n_local_s,n_global_s,
                                            reward,terminated])


    
    def decay_epsilon(self,frame):
        """ Used to decay the epsilon down for the greedy policy"""
        start_epsilon = self.start_epsilon
        final_epsilon = self.final_epsilon
       
        #self.epsilon = max(final_epsilon,start_epsilon + ((final_epsilon-start_epsilon)/n_episodes)*episode )
        #self.epsilon = max(final_epsilon,self.epsilon * (0.9995))
        # Linear decline of the epsilon decay
        linear_decay = start_epsilon*((self.decay_stop - frame)/self.decay_stop) + final_epsilon*(frame/self.decay_stop)
        self.epsilon = max(final_epsilon,linear_decay)

    def setup_environment(self, maze, agent_pos, target_pos,start_dist):
        env = gym.make('Maze_env/MazeRunner-v0',len_game= self.len_game,
                        num_agents=self.n_agents,vision_len=self.agents.vision
                            ,maze=maze, render_mode='rgb_array',obs_type = 'spatial',
                            action_type=self.agents.action_type,
                            agents_pos = agent_pos, 
                            targets_pos = target_pos,
                            start_dist = self.start_dist)
        
        # --- environment wrappers --- #
        #env = gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_episodes)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = sA.maze_runner_stickyActions(env,self.n_agents,sticky_prob=0.25)
        env = self.agents.add_wrappers(env)

        return env
    
    def update_networks(self,update_start,frame):
                # --- soft update of target Q-net --- #
                if update_start and frame % self.target_update ==0:
                    
                    self.soft_update(tau=self.tau)
                
                # --- update policy Q-net --- #
                if update_start and frame % self.policy_update == 0:
                    
                    # -- zero out gradients --- #
                    self.optimizer.zero_grad()
                    
                    # --- compute loss --- #
                    loss,action_prob = self.compute_loss(frame)

                    # --- save losses --- #
                    self.losses.append(loss.detach().cpu().numpy())

                    # --- enact backpropogation --- #
                    loss.backward()

                    # --- cutoff gradients --- #
                    #nn.utils.clip_grad_norm_(self.agents.Q_fun.parameters(),1)

                    # --- step optimizer --- #
                    self.optimizer.step()

                    # --- schedular step --- #
                    self.scheduler.step()

                    if frame % 10000 ==0:
                        print(f'frame [{frame}:{self.n_frames}] with loss {loss}')
                        for param_group in self.optimizer.param_groups:
                            print(f'Learning rate : {param_group["lr"]}')
                        print(f'Epsilon: {self.epsilonScheduler.epsilon}')
                        print(f'Epsilon Level: {self.epsilonScheduler.cur_level}')
                        print(f'Start dist: {self.start_dist}')
                        

                # --- decay the epsilon greedy policy --- #   
                #self.decay_epsilon(frame)
                if update_start:
                    if self.curriculum :
                        updated = self.epsilonScheduler.step()
                        
                        if updated['dist']:
                            # -- increase the starting distance -- #
                            self.start_dist+=1
                            print(f'Increasing Distance to {self.start_dist}')
                            self.distance_upgrades.append(frame)
                        if updated['level']:
                            # -- remember that we upgraded the level -- #
                            self.epsilon_upgrades.append(frame)
                            # -- scale the learning rate back -- #
                            for param_group in self.optimizer.param_groups:
                                param_group['lr']*= np.power(1.01,1)
                    else:
                        self.epsilonScheduler.step()
                self.replay_buffer.step(frame,self.n_frames)

    def get_action(self,env,state,cur_dist=None):

        # -- levels of the epsilon -- #
        epsilon = self.epsilonScheduler.epsilon
        actions = []
        # --- get action --- #
        for a in range(self.n_agents):
            dist_epsilon = 1.0
            if self.curriculum:
                
                if cur_dist[a]<=self.epsilonScheduler.cur_level:
                    dist_epsilon = epsilon[cur_dist[a]-1]  
            else:
                dist_epsilon = epsilon[0]
            
            actions.append(self.agents.get_single_agent_action(
                    env = env,
                    state=state,
                    a=a,
                    epsilon = dist_epsilon
                ))

        # --- save actions for results --- #
        self.actions_taken.append(actions)

        return actions


    def train(self, test_agent = False, peak = False, uniform_loc = False):
        """ Train the agents in maze runner
            test_agent: outputs a test of the agent to the user"""

        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(self.n_agents+2,2)
        
        #--- initialize random maze --- #
        random_index = random.randint(0,len(self.mazes)-1)
        maze = self.mazes[random_index]

        # -- uniform training parameter -- #
        uniform_training = False


        # --- this is used to make sure that the agent goes through every scenario --- #
        if uniform_loc and self.agent_pos is None and self.target_pos is None:
            total_spots = self.mazes.shape[0]*self.mazes.shape[1]
            uniform_training = True

            # -- make the sequence of agent and target positions -- #
            spots = [(i,j) for i in range(total_spots) for j in range(total_spots) if i!=j]
            se = 0

            # --- maze environment --- #
            env = self.setup_environment(maze,None,None,None)

        else:
            # --- maze environment --- #
            env = self.setup_environment(maze,
                                         self.agent_pos,
                                         self.target_pos,
                                         start_dist = self.start_dist)
        
        self.agents.Q_fun.train()
        frame = 0
        ep = 0
        start_updating = False
        stop_training = False

        while frame < self.n_frames:

            # --- initalize another random maze --- #
            if ep % MAZE_UPDATE == 0:
                random_index = random.randint(0,len(self.mazes)-1)
                maze = self.mazes[random_index]

            # --- reset the environment --- #
            if uniform_training:
                state,info = env.reset(options = {'new_maze': maze,
                                                  'agent_pos': [spots[se][0]],
                                                  'target_pos': [spots[se][1]],
                                                })
                done = False
                if se < len(spots)-1:
                    se+=1
                else:
                    se = 0
            else:
                state,info = env.reset(options = {'new_maze': maze,
                                                  'start_dist':self.start_dist})
                done = False

            if self.curriculum:
                # -- current position -- #
                cur_dist = []
                for a in range(self.n_agents):
                    cur_dist.append(info[f'agent_{a}']['man_dist'])
            else:
                cur_dist = None
            # --- cumulitive episode reward --- #
            cum_reward = [0 for _ in range(self.n_agents)]

            # --- start episode --- #
            while not done:
                 # -- get action -- #
                action = self.get_action(env,state, cur_dist)

                # --- get next state and rewards from this action --- #
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # --- save each of the agents state, rewards, ect.. --- #
                for a in range(self.n_agents):

                    # --- add experience to replay buffer --- #
                    self.append_to_RB(state[f'local_{a}'],state[f'global_{a}'],action[a],
                                            next_state[f'local_{a}'],next_state[f'global_{a}'],
                                            reward[a],terminated,a)
                    
                    # --- accumulate rewards --- #
                    cum_reward[a] += reward[a]
                    
                    # -- save agents dist -- #
                    if self.curriculum:
                        cur_dist[a] = info[f'agent_{a}']['man_dist']

                # --- record agent's rewards --- #
                for a in range(self.n_agents):
                    self.cum_reward[f'agent_{a}'].append(cum_reward[a])

                # -- next state --- #
                state = next_state

                # --- processes if episode is done --- #
                done = truncated or terminated

                # --- replay buffer is filled up to a point --- #
                if len(self.replay_buffer)>=int(self.replay_buffer_size * self.replay_buffer_min_perc):
                    start_updating = True

                self.update_networks(start_updating,frame)
                

                frame+=1

            env.close() 
            ep +=1  


            # -- test how well the agent is doing -- #
            if start_updating and ep % 500 == 0:
                success_rate = self.agents.test_agent(self.mazes,
                                                    n_episodes = 500,
                                                    num_agents = self.n_agents,
                                                    len_game = 15,
                                                    start_dist=self.start_dist)
                print(f'Current Score: {success_rate}')
                self.scores.append([frame,success_rate])
                
                if self.epsilonScheduler.check_threshold(success_rate) and False:
                    self.epsilonScheduler.reset(start_epsilon = 0.5)

                if success_rate > self.best_score:
                    self.best_agent = self.agents.copy()
                    self.best_score = success_rate

                if success_rate >=0.99:
                    print(f'Agent has gotten to success rate {success_rate}.  Stopping training')
                    stop_training=True

            # -- here we have our during training functions -- #
            if test_agent and start_updating and ep % 1000 == 0:
                self.agents.run_agent(maze,
                                      len_game = 15,
                                      n_episodes = 5,
                                      num_agents = self.n_agents,
                                      epsilon=0,
                                      agents_pos = self.agent_pos,
                                      targets_pos = self.target_pos,
                                      start_dist = self.epsilonScheduler.cur_level) 
            # -- update the plots -- #
            if peak and ep % 500 == 0 and start_updating:
                # Update the plot
                self.update_plots(frame,fig,ax)
                plt.pause(0.1) 
            
            if stop_training:
                break
                
        plt.ioff()  
        plt.show()

    def update_plots(self, frame,fig, axe):
        clear_output(wait=True)
        window_size = int(frame*0.01)
        losses_series = pd.Series(self.losses)
        moving_avg_losses = losses_series.rolling(window=window_size).mean()
        
        axe[0][0].cla()
        axe[0][0].plot(moving_avg_losses)
        axe[0][0].set_xlabel('frame')
        axe[0][0].set_ylabel('loss')
        axe[0][0].set_title('losses')

        actions_taken = np.array(self.actions_taken)
        actions_taken = actions_taken.flatten()
        axe[0][1].cla()
        axe[0][1].hist(actions_taken)
        axe[0][1].set_title('histogram of actions')

        # --- Q-values --- #
        moving_avg_q = {}
        for a in range(self.agents.n_actions):
            action_series = pd.Series(self.Q_values[a])
            moving_avg_action_temp = action_series.rolling(window_size).mean()
            moving_avg_q[a] = moving_avg_action_temp

        data = []
        for action, q_vals in moving_avg_q.items():
            for frame, q in enumerate(q_vals):
                data.append({"frame": frame, "Q-Value": q, "Action": action})

        df = pd.DataFrame(data)
        
        # Plot using Seaborn
        axe[1][0].cla()
        sns.lineplot(data=df, x="frame", y="Q-Value", hue="Action", ax=axe[1][0], palette="tab10")


        # --- line plot of scores --- #
        axe[1][1].cla()
        scores_data = []
        for frame, score in self.scores:
            scores_data.append({'frame': frame, 'score':score})
        scores_df = pd.DataFrame(scores_data)
        sns.lineplot(data = scores_df, x ='frame',y = 'score',ax = axe[1][1],palette='tab10')


        # -- if curriculum was used, add in extra features to the pictures -- #
        if self.curriculum:
            for frame in self.epsilon_upgrades:
                axe[0][0].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
                axe[1][0].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
                axe[1][1].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
                
            for frame in self.distance_upgrades:
                axe[0][0].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')
                axe[1][0].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')
                axe[1][1].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')

        for a in range(2,self.n_agents+2):
            # Lets find a moving average of the scores
              # Adjust based on how much smoothing you want
            scores_series = pd.Series(self.cum_reward[f'agent_{a-2}'])
            moving_avg_reward = scores_series.rolling(window=window_size).mean()

            axe[a][0].cla()
            axe[a][0].plot(moving_avg_reward)
            axe[a][0].set_xlabel('frame')
            axe[a][0].set_ylabel('cum awards')
            axe[a][0].set_title('cumulative awards across episodes')

            td_series = pd.Series(self.td_errors[a-2])
            moving_avg_td = td_series.rolling(window=window_size).mean()

            axe[a][1].cla()
            axe[a][1].plot(moving_avg_td)
            axe[a][1].set_xlabel('frame')
            axe[a][1].set_ylabel('td error')
            axe[a][1].set_title('error between target and policy')

            if self.curriculum:
                for frame in self.epsilon_upgrades:
                    axe[a][0].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
                    axe[a][1].axvline(x=frame,linestyle = 'dashed', color = 'blue',alpha=0.8,label = 'epsilon upgraded')
                    
                for frame in self.distance_upgrades:
                    axe[a][0].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')
                    axe[a][1].axvline(x = frame,linestyle = 'dashed',color = 'red',alpha=0.8,label = 'distance upgraded')


        


        display(fig)

    def results(self):
        """ Output the losses, action distribution, running avg of Q-values,
        and cumulative rewards for each agent"""
        fig, axe = plt.subplots(self.n_agents+2,2,figsize=(10,10))
        
        self.update_plots(self.n_frames,fig,axe)


        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')

        if os.path.exists(fd)==False:
            os.mkdir(fd)

        fd_original = os.path.join(fd,f'{self.name}')
        if os.path.exists(fd_original)==False:
            os.mkdir(fd_original)

        fd_best = os.path.join(fd,f'{self.name}_best')
        if os.path.exists(fd_best)==False:
            os.mkdir(fd_best)
        
        
        plt.savefig(os.path.join(fd_original,'results.png'))
        plt.savefig(os.path.join(fd_best,'results.png'))


    def dist_rewards(self,dist):
        """ Output the distribution of rewards, need the rewards wrapper to work"""
        keys = list(dist.keys())
        values = list(dist.values())
        plt.bar(keys,values)
        plt.savefig(os.path.join(self.filepath,'rewards_dist.png'))

    def save_checkpoint(self, episode):
        """Save checkpoint if training stops"""
        if os.path.exists(self.filepath)==False:
            os.mkdir(self.filepath)

        training_checkpoint = {
            'agent' : self.agents.Q_fun.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
            'n_episodes' : self.n_episodes,
            'scheduler' : self.scheduler.state_dict(),
            'losses': self.losses,
            'cum_rewards': self.cum_reward,
            'actions_taken' : self.actions_taken,
            'replay_buffer': self.replay_buffer,
            'epsilon' : self.epsilon,
            'final_epsilon': self.final_epsilon,
        }
        torch.save(training_checkpoint, os.path.join(self.filepath, 'training.pth'))

    def save(self):
        """Save the model"""
        # --- first save agent model --- #
        self.agents.save(self.name)

        # --- next, save the best agent model --- #
        self.best_agent.save(self.name + '_best')

        # --- next save the hyperparameters --- #
        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')

        if os.path.exists(fd)==False:
            os.mkdir(fd)
        # -- final agent from the training -- #
        fd_original = os.path.join(fd,f'{self.name}')
        if os.path.exists(fd_original)==False:
            os.mkdir(fd_original)
        # -- best agent from the training -- #
        fd_best = os.path.join(fd,f'{self.name}_best')
        if os.path.exists(fd_best)==False:
            os.mkdir(fd_best)

        # now to save the hyperparameters for this mod
        param = {
            'len_game': self.len_game,
            'n_agents': self.n_agents,
            'replay_buffer_size': self.replay_buffer_size,
            'policy_update': self.policy_update,
            'target_update': self.target_update,
            'gamma': self.gamma,
            'tau' : self.tau,
            'batch_size': self.batch_size,
            'lambda_entropy': self.lambda_entropy,
            'lr': self.lr,
            'lr_step_size': self.lr_step_size,
            'lr_gamma': self.lr_gamma,
            'n_frames': self.n_frames,
            'alpha': self.alpha,
            'beta' : self.beta,
            'decay_total': self.decay_total,
            'per' : self.per,
            'agent_pos' : self.agent_pos,
            'target_pos' : self.target_pos,
            'curriculum': self.curriculum,
            'curriculum_mu': self.epsilonScheduler.mu,
            'curriculum_alpha': self.epsilonScheduler.alpha,
        }

        with open(os.path.join(fd_original,'hyperparameters.json'),'w') as f:
            json.dump(param,f,indent=4)
        
        with open(os.path.join(fd_best,'hyperparameters.json'),'w') as f:
            json.dump(param,f,indent=4)
        

    def load(self):
        """Load a pre-trained model for more training"""
        
        checkpoint = torch.load(os.path.join(self.filepath,'training.pth'))
        

        self.agents.Q_fun.load_state_dict(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_episodes = checkpoint['episode']
        self.n_episodes = checkpoint['n_episodes']
        self.losses = checkpoint['losses']
        self.cum_reward = checkpoint['cum_rewards']
        self.actions_taken = checkpoint['dis']
        self.replay_buffer = checkpoint['replay_buffer']

    