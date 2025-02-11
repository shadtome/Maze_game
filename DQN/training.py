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


class Maze_Training:
    def __init__(self,name,maze_dataset,len_game = 1000,n_agents=1,vision=1, action_type = 'full',
                 lr=1e-3,start_epsilon=1,final_epsilon=0.1, n_episodes=100, update_factor=500):
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

        # The agent with the Q_function
        self.agents = agent.CNN_Maze_Agents(vision,action_type)

        # The target Q_net
        self.target_Q_net = self.agents.Q_fun.copy()
        self.target_Q_net = self.target_Q_net.to(device=device.DEVICE)

        # Collection of mazes (can be single maze or multiple)
        self.mazes = maze_dataset

        # Number of agents
        self.n_agents = n_agents


        # max length of each episode
        self.len_game = len_game

        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)

        # file name for saving and loading
        self.name = name

        # Where to start and end the number of episodes
        self.start_episodes = 0
        self.n_episodes = n_episodes

        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon

        # Learning Rate
        self.lr = lr

        # Used for training
        #self.loss_fun = nn.MSELoss()
        self.loss_fun = nn.SmoothL1Loss()  # The Huber loss function, more stable to outliers

        # Optimizer
        self.optimizer = torch.optim.Adam(self.agents.Q_fun.parameters(),lr=self.lr)

        # Schedular to lower the lr
        self.scheduler = StepLR(self.optimizer,step_size=100,gamma=0.5)

        # results from training
        self.losses = []
        self.cum_reward = {}
        self.update_factor = update_factor
        self.actions_taken = []

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

    def compute_loss(self,batch_size, gamma=0.99,lambda_entropy = 0.01):
        """This is to compute the loss between the target Q-network and the policy Q-network.
        This also adds a entropy term.  The entropy terms wants to be maximized so that it is not
        determinisitic, i.e., staying in one spot forever.
        
        gamma: a regularizing term that ensures convergence a value in (0,1)
        lambda_entropy: this is the regularization term for the entropy loss, here 
                        we want to maximize entropy."""
        
        # Get the states and actions
        local_s,global_s, actions,next_local_s,next_global_s, rewards, terminated = self.sample_replay(batch_size)
        
        # get our policy q-values 
        q_values = self.agents.Q_fun(local_s,global_s)
        selected_q_values = q_values.gather(1,actions.unsqueeze(1)).squeeze(1)
        #print('q_values',torch.isinf(q_values).any())
        with torch.no_grad():
            # Next, we calculate the the target q-values which are maximized
            next_q_values = self.target_Q_net(next_local_s,next_global_s).max(1)[0]

            # Then we calculate the estimated Bellmann expectation operator
            target = rewards + gamma*next_q_values * (1-terminated.float())

        # Compute the loss between the Bellman expectation operator and the policy q-values
        loss = self.loss_fun(selected_q_values,target)

        # Now, for the entropy, we want to look at $pi(a|s), the probablity of action a given state s
        # and compute its entropy over all the actions, and take the average over the states.
        action_probs = torch.softmax(q_values,dim=1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-6),dim=1)
        #print('entropy',torch.isinf(entropy).any())
        loss -=lambda_entropy * entropy.mean()
        #print('Loss',torch.isinf(loss).any())
        return loss, action_probs
    
    def decay_epsilon(self,episode,n_episodes):
        """ Used to decay the epsilon down for the greedy policy"""
        start_epsilon = self.start_epsilon
        final_epsilon = self.final_epsilon
       
        #self.epsilon = max(final_epsilon,start_epsilon + ((final_epsilon-start_epsilon)/n_episodes)*episode )
        self.epsilon = max(final_epsilon,self.epsilon * (0.9995))

    def train(self,lambda_entropy = 0.01):
        """The deep Q-network training for our maze runner"""

        # set up some initial variables for training
        n_episodes = self.n_episodes

        # discount factor
        gamma = 0.99

        # This is used to change the maze enviroment to train in, so that the agent 
        # does not hyper learn one maze
        n_epi_per_maze = 100
       
        # get a random initial maze
        random_index = random.randint(0,len(self.mazes)-1)
        maze = self.mazes[random_index]

        # set up our maze enviroment for spatial learning
        env = gym.make('Maze_env/MazeRunner-v0',len_game= self.len_game,
                       num_agents=self.n_agents,vision_len=self.agents.vision
                        ,maze=maze, render_mode='rgb_array',obs_type = 'spatial',
                        action_type=self.agents.action_type)
        
        # set up the statistics wrapper for the enviroment and the user wrappers
        #env = gym.wrappers.RecordEpisodeStatistics(env,buffer_length=n_episodes)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = self.agents.add_wrappers(env)
        #env = gym.wrappers.NormalizeObservation(env)
            
        
        update_target = 0
        for ep in range(self.start_episodes,n_episodes):
            
            if ep % n_epi_per_maze == 0:
                random_index = random.randint(0,len(self.mazes)-1)
                maze = self.mazes[random_index]

            state,info = env.reset(options = {'new_maze': maze})
            #print(f'episode {ep}')
            done = False

            cum_reward = [0 for _ in range(self.n_agents)]


            #self.target_Q_net.load_state_dict(self.agents.Q_fun.state_dict())

            #for param in self.agents.Q_fun.parameters():
                #if param.grad is not None:
                    #print(param.grad)

            
            while not done:
                #######################
                # Get the action corresponding to the state
                action = self.agents.get_action(env,self.n_agents,state,self.epsilon)

                self.actions_taken.append(action)

                #########################
                # For each action, get the next state for each agent
                next_state, reward, terminated, truncated, info = env.step(action)
                ########################
                # Put the states, actions and next states in to the replay buffer for training
                for a in range(self.n_agents):
                    self.replay_buffer.append([state[f'local_{a}'],state[f'global_{a}'],action[a],
                                               next_state[f'local_{a}'],next_state[f'global_{a}'],
                                               reward[a],terminated])
                    cum_reward[a] += reward[a]

                state = next_state

                # This gets if all the agents go to their finish line or passes the threshold
                # of the length of the game
                done = truncated or terminated
                
                #######################
                # After the replay is big enough we have gone through enough 
                # iterations of the enviroment, enact a soft update of the target Q-net
                if len(self.replay_buffer)>=10000 and update_target % self.update_factor ==0:
                    #self.target_Q_net.load_state_dict(self.agent.Q_fun.state_dict())
                    self.soft_update(tau=0.01)

                update_target+=1


                ########################
                # Enact stochastic Gradient Descent 
                if len(self.replay_buffer)>=10000 and update_target % 4 == 0:

                    self.optimizer.zero_grad()
                    loss,action_prob = self.compute_loss(batch_size=64,lambda_entropy=lambda_entropy)
                    self.losses.append(loss.detach().cpu().numpy())
                    loss.backward()

                    # Implement gradient cutoffs
                    nn.utils.clip_grad_norm_(self.agents.Q_fun.parameters(),1)

                    self.optimizer.step()
                    self.scheduler.step()
                    if update_target % 100 ==0:
                        print(f'episode {ep} with loss {loss}')
                    #if update_target % 200 == 0:
                        #print(f'action|state distribution: {action_prob.mean(0)}')
                        #print(f'action|state probablities: {action_prob[0]}\n {action_prob[-1]}')
                        
                    self.decay_epsilon(ep,n_episodes)
            # Recoard the cumulative reward for the agents
            for a in range(self.n_agents):
                self.cum_reward[f'agent_{a}'].append(cum_reward[a])
            #print(f'cumulative reward: {cum_reward}')
            
            

            #checkpoint save:
            #if ep %20==0:
                #self.save_checkpoint(ep)
        #self.dist_rewards(env.dist_rewards)
        env.close()
                
    def results(self):
        """ This is used to print the losses over the training, 
                and the distribution of actions.  Important for 
                seeing if the agents are focusing too much on a action"""
        fig, axe = plt.subplots(self.n_agents+1,2,figsize=(10,10))
        
        axe[0][0].plot(self.losses)
        axe[0][0].set_xlabel('episode')
        axe[0][0].set_ylabel('loss')
        axe[0][0].set_title('losses')

        actions_taken = np.array(self.actions_taken)
        actions_taken = actions_taken.flatten()
        axe[0][1].hist(actions_taken)
        axe[0][1].set_title('histogram of actions')

        for a in range(1,self.n_agents+1):
            axe[a][0].plot(self.cum_reward[f'agent_{a-1}'])
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

    