import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import gymnasium as gym
import DQN.agent as agent
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import os
import device
import pandas as pd



#################
# ----- Fixed Hyperparameters ---------- #
REPLAY_BUFFER_SIZE = 100000
MAZE_UPDATE = 100
POLICY_UPDATE = 4
TARGET_UPDATE = 10000




class Maze_Training:
    def __init__(self,name,maze_dataset,len_game = 1000,
                 n_agents=1,vision=1, action_type = 'full',
                 gamma=0.99,tau = 0.01, batch_size=32, lambda_entropy = 0.01, 
                 lr=1e-3,start_epsilon=1,final_epsilon=0.1,n_frames=50000):
        """Used to train a deep Q-network for agents exploring a maze.

        maze_dataset: a dataset of mazes that come from the maze_dataset package

        n_agents: the maze enviroment can run multiple agents and you can accumulate their 
        experiences together.  But it is aimed to be adversarial in the sense we want to avoid
        others

        start_epsilon: this is the start epsilon for the greedy policy that is enacted and decreased
        as the agent gets better.

        final_epsilon: this is the final epsilon that it will decay to.

        n_episodes: number of iterations of the enviroment the agent will go through while training

        update_factor: This how many times the target Will change this"""


        # --- Dynamic Hyperparameters --- #
        self.gamma = gamma 
        self.tau = tau   
        self.batch_size = batch_size 
        self.lambda_entropy = lambda_entropy


        # The agent with the Q_function
        self.agents = agent.CNN_Maze_Agents(vision,action_type)

        # The target Q_net
        self.target_Q_net = self.agents.Q_fun.copy()
        self.target_Q_net = self.target_Q_net.to(device=device.DEVICE)
        self.target_Q_net.eval()

        # Collection of mazes (can be single maze or multiple)
        self.mazes = maze_dataset

        # Number of agents
        self.n_agents = n_agents


        # max length of each episode
        self.len_game = len_game

        # Replay buffer
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        # file name for saving and loading
        self.name = name

        # Where to start and end the number of episodes
        self.n_frames = n_frames

        # --- epsilon-greedy policy --- #
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon

        # Learning Rate
        self.lr = lr

        # Used for training
        self.loss_fun = nn.MSELoss()
        #self.loss_fun = nn.SmoothL1Loss()  # The Huber loss function, more stable to outliers

        # Optimizer
        self.optimizer = torch.optim.Adam(self.agents.Q_fun.parameters(),lr=self.lr,amsgrad=False)
        #self.optimizer = torch.optim.RMSprop(self.agents.Q_fun.parameters(),lr=lr)

        # Schedular to lower the lr
        self.scheduler = StepLR(self.optimizer,step_size=10,gamma=0.1)

        # results from training
        self.losses = []
        self.cum_reward = {}
        self.actions_taken = []
        self.Q_values = []

        for a in range(self.n_agents):
            self.cum_reward[f'agent_{a}'] = []


    def soft_update(self,tau):
        """ Used to update the current target Q-network by the policy Q-network by a weighted sum:
            i.e., target = tau*policy + (1-tau)*target.  This ensures that the training is smoother:
            
            tau: this is a number in [0,1], where tau=1 implies that we replace the target parameters
            with the policy parameters.  We use a small tau to make the policy slowly update the target."""
        
        for target_param, policy_param in zip(self.target_Q_net.parameters(),self.agents.Q_fun.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def sample_replay(self, batch_size):
        """ Used to sample a collection of experiences fro the replay buffer that the agents have done:
        i.e., it is filled with initial state, action take, then the resulting state with corresponding reward"""

        # sample from the buffer
        sample = random.sample(self.replay_buffer,batch_size)
        # get the information
        local_state,global_state, action, next_local_state, next_global_state, reward,terminated = zip(*sample)

        # we need everything to be a tensor to put into our neural network,
        # furthermore, we will need to permute our shape, since gymnasium outputs images
        # as (h,w,c)
        local_st = torch.tensor(np.array(local_state),dtype=torch.float32,device = device.DEVICE)
        local_st = local_st.permute((0,3,1,2))
        local_st = local_st/255

        global_st = torch.tensor(np.array(global_state),dtype=torch.float32,device = device.DEVICE)
        
        global_st[:,0:2] = global_st[:,0:2]/(self.agents.CNN_shape[1]*self.agents.CNN_shape[2]-1)
        global_st[:,3] = global_st[:,3]/(self.agents.CNN_shape[1] + self.agents.CNN_shape[2])

        action_t = torch.tensor(np.array(action), dtype=torch.int64,device = device.DEVICE)

        next_local_st = torch.tensor(np.array(next_local_state),dtype=torch.float32,device=device.DEVICE)
        next_local_st = next_local_st.permute((0,3,1,2))

        next_global_st = torch.tensor(np.array(next_global_state),dtype=torch.float32,device = device.DEVICE)

        reward_t = torch.tensor(np.array(reward),dtype=torch.float32,device = device.DEVICE)
        terminated_t = torch.tensor(np.array(terminated),dtype = torch.int64,device= device.DEVICE)


        return local_st,global_st,action_t,next_local_st,next_global_st,reward_t, terminated_t

    def compute_loss(self):
        """This is to compute the loss between the target Q-network and the policy Q-network.
        This also adds a entropy term.  The entropy terms wants to be maximized so that it is not
        determinisitic, i.e., staying in one spot forever.
        
        gamma: a regularizing term that ensures convergence a value in (0,1)
        lambda_entropy: this is the regularization term for the entropy loss, here 
                        we want to maximize entropy."""
        
        # Get the states and actions
        local_s,global_s, actions,next_local_s,next_global_s, rewards, terminated = self.sample_replay(self.batch_size)
        
        # We need to find $Q^*(s,a) \approx r + \gamma * Q(s', max_{a'} Q'(s',a'))
        # Where (s,a,r,s') is from the replay buffer, Q is the policy net, Q' is the target net


        # get Q(s,a)
        q_values = self.agents.Q_fun(local_s,global_s)
        selected_q_values = q_values.gather(1,actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            next_actions = self.agents.Q_fun(next_local_s, next_global_s).argmax(1, keepdim=True)

            # Next, we calculate the the target q-values which are maximized
            next_q_values = self.target_Q_net(next_local_s,next_global_s).gather(1,next_actions).squeeze(1)
            
            # Then we calculate the estimated Bellmann expectation operator
            target = rewards + self.gamma*next_q_values * (1-terminated.float())

        # Compute the loss between the Bellman expectation operator and the policy q-values
        loss = self.loss_fun(selected_q_values,target)
        #print(f"Reward mean: {rewards.mean().item()}, std: {rewards.std().item()}")
        #print(f"Q_target mean: {target.mean().item()}, std: {target.std().item()}")
        #print(f"Q_policy mean: {selected_q_values.mean().item()}, std: {selected_q_values.std().item()}")
        
        # Now, for the entropy, we want to look at $pi(a|s), the probablity of action a given state s
        # and compute its entropy over all the actions, and take the average over the states.
        action_probs = torch.softmax(q_values,dim=1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-6),dim=1)
        
        #print('entropy',torch.isinf(entropy).any())
        loss -=self.lambda_entropy * entropy.mean()
        #print('Loss',torch.isinf(loss).any())
        return loss, action_probs
    
    def decay_epsilon(self,frame):
        """ Used to decay the epsilon down for the greedy policy"""
        start_epsilon = self.start_epsilon
        final_epsilon = self.final_epsilon
       
        #self.epsilon = max(final_epsilon,start_epsilon + ((final_epsilon-start_epsilon)/n_episodes)*episode )
        #self.epsilon = max(final_epsilon,self.epsilon * (0.9995))
        # Linear decline of the epsilon decay
        linear_decay = start_epsilon*((self.n_frames - frame)/self.n_frames) + final_epsilon*(frame/self.n_frames)
        self.epsilon = max(final_epsilon,linear_decay)


    def train(self):
        """ Train the agents in maze runner"""

        #--- initialize random maze --- #
        random_index = random.randint(0,len(self.mazes)-1)
        maze = self.mazes[random_index]


        # --- maze environment --- #
        env = gym.make('Maze_env/MazeRunner-v0',len_game= self.len_game,
                       num_agents=self.n_agents,vision_len=self.agents.vision
                        ,maze=maze, render_mode='rgb_array',obs_type = 'spatial',
                        action_type=self.agents.action_type)
        
        # --- environment wrappers --- #
        #env = gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_episodes)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = self.agents.add_wrappers(env)
        
        frame = 0
        ep = 0
        while frame < self.n_frames:

            # --- initalize another random maze --- #
            if ep % MAZE_UPDATE == 0:
                random_index = random.randint(0,len(self.mazes)-1)
                maze = self.mazes[random_index]

            # --- reset the environment --- #
            state,info = env.reset(options = {'new_maze': maze})
            
            done = False

            # --- cumulitive episode reward --- #
            cum_reward = [0 for _ in range(self.n_agents)]

            while not done:

                # --- get action --- #
                action = self.agents.get_action(env,self.n_agents,state,self.epsilon)

                # --- save actions for results --- #
                self.actions_taken.append(action)

                
                # --- get next state and rewards from this action --- #
                next_state, reward, terminated, truncated, info = env.step(action)

                # --- save each of the agents state, rewards, ect.. --- #
                for a in range(self.n_agents):
                    self.replay_buffer.append([state[f'local_{a}'],state[f'global_{a}'],action[a],
                                            next_state[f'local_{a}'],next_state[f'global_{a}'],
                                            reward[a],terminated])
                    # --- accumulate rewards --- #
                    cum_reward[a] += reward[a]

                # --- record agent's rewards --- #
                for a in range(self.n_agents):
                    self.cum_reward[f'agent_{a}'].append(cum_reward[a])

                # -- next state --- #
                state = next_state

                # --- processes if episode is done --- #
                done = truncated or terminated


                # --- soft update of target Q-net --- #
                if len(self.replay_buffer)>=int(REPLAY_BUFFER_SIZE/10) and frame % TARGET_UPDATE ==0:
                    
                    self.soft_update(tau=self.tau)

                # --- update policy Q-net --- #
                if len(self.replay_buffer)>=int(REPLAY_BUFFER_SIZE/10) and frame % POLICY_UPDATE == 0:
                    # -- zero out gradients --- #
                    self.optimizer.zero_grad()
                    
                    # --- compute loss --- #
                    loss,action_prob = self.compute_loss()

                    # --- save losses --- #
                    self.losses.append(loss.detach().cpu().numpy())

                    # --- enact backpropogation --- #
                    loss.backward()

                    # --- cutoff gradients --- #
                    nn.utils.clip_grad_norm_(self.agents.Q_fun.parameters(),1)

                    # --- step optimizer --- #
                    self.optimizer.step()

                    # --- schedular step --- #
                    #self.scheduler.step()

                    if frame % 10000 ==0:
                        print(f'frame {frame} with loss {loss}')
                

                # --- decay the epsilon greedy policy --- #   
                self.decay_epsilon(frame)

                frame+=1

            env.close() 
            ep +=1    

                
    def results(self):
        """ This is used to print the losses over the training, 
                and the distribution of actions.  Important for 
                seeing if the agents are focusing too much on a action"""
        fig, axe = plt.subplots(self.n_agents+1,2,figsize=(10,10))
        window_size = 10
        losses_series = pd.Series(self.losses)
        moving_avg_losses = losses_series.rolling(window=window_size).mean()
        
        axe[0][0].plot(moving_avg_losses)
        axe[0][0].set_xlabel('episode')
        axe[0][0].set_ylabel('loss')
        axe[0][0].set_title('losses')

        actions_taken = np.array(self.actions_taken)
        actions_taken = actions_taken.flatten()
        axe[0][1].hist(actions_taken)
        axe[0][1].set_title('histogram of actions')


        for a in range(1,self.n_agents+1):
            # Lets find a moving average of the scores
              # Adjust based on how much smoothing you want
            scores_series = pd.Series(self.cum_reward[f'agent_{a-1}'])
            moving_avg_reward = scores_series.rolling(window=window_size).mean()

            axe[a][0].plot(moving_avg_reward)
            axe[a][0].set_xlabel('episode')
            axe[a][0].set_ylabel('cum awards')
            axe[a][0].set_title('cumulative awards across episodes')

        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')

        if os.path.exists(fd)==False:
            os.mkdir(fd)

        fd = os.path.join(fd,f'{self.name}')
        if os.path.exists(fd)==False:
            os.mkdir(fd)
        
        plt.savefig(os.path.join(fd,'results.png'))

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
        fd = os.getcwd()
        fd = os.path.join(fd,'trained_agents')

        if os.path.exists(fd)==False:
            os.mkdir(fd)

        fd = os.path.join(fd,f'{self.name}')
        if os.path.exists(fd)==False:
            os.mkdir(fd)

        torch.save(self.agents.Q_fun.state_dict(),os.path.join(fd,'agent.pth'))
        

    def load(self):
        """Load the model for furhter training"""
        
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

    