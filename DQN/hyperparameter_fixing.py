from DQN.models import base
from DQN.agents import basic
from DQN.training import basic
from Maze_env.wrappers.reward_wrappers.runner_rewards import reward_dist
import random


class agent_tuning:
    def __init__(self,name, n_iter, maze_dataset,model,vision,reward, **kwargs):
        
        # -- best built agents in hyperparameter tuning -- #
        self.best_training = None
        self.best_score = 0

        # -- reward distribution -- #
        self.rewards = reward

        # -- fixed parameters -- #
        self.fix_param = kwargs

        # -- parameter ranges -- #
        self.param = {'len_game': [15, 30, 75, 100],
                      'gamma' : [0.8, 0.95, 0.99],
                      'tau': [0.1, 0.01, 0.005, 0.001],
                      'n_frames': [100000,500000,1000000],
                      'batch_size': [32,64,128],
                      'lr': [0.01,0.001,0.0001, 0.00001],
                      'lr_step_size': [1000, 3000, 10000],
                      'lr_gamma': [0.9,0.8,0.1],
                      'replay_buffer_size': [10000,50000,100000],
                      'replay_buffer_min_perc': [0.1,0.3,0.5],
                      'target_update': [500,1000,2000,5000],
                      'policy_update':[1,2,4, 6],
                      'lambda_entropy': [0.0,0.1,0.2,0.4,0.6],
                      'alpha' : [0.2,0.5,0.6],
                      'beta': [0.4],
                      'decay_total':[100,1000,10000,50000,100000],
                      'curcurriculum' : [True, False]
        }

        # -- check if any of the parameters are fixed -- #
        for k in self.fix_param:
            self.param[k] = [self.fix_param[k]]

        self.tune(name,n_iter, model,vision,maze_dataset)
        self.best_training.save()
        self.best_training.results()

    def tune(self,name, n_iter, model, vision, maze_dataset):

        for i in range(n_iter):
            maze_agent = basic.maze_agents(model,
                               vision=vision,
                               action_type='cardinal',
                               rewards_dist=self.rewards)
            
            train = basic.Maze_Training(name = name,
                              maze_dataset = maze_dataset,
                              maze_agent = maze_agent,
                              len_game=random.sample(self.param['len_game'],k=1)[0],
                              n_agents=1,
                              gamma = random.sample(self.param['gamma'],1)[0],
                              tau = random.sample(self.param['tau'],1)[0],
                               batch_size = random.sample(self.param['batch_size'],1)[0],
                              n_frames = random.sample(self.param['n_frames'],1)[0],
                              lr = random.sample(self.param['lr'],1)[0],
                              lr_step_size=random.sample(self.param['lr_step_size'],1)[0],
                              lr_gamma = random.sample(self.param['lr_gamma'],1)[0],
                              replay_buffer_size=random.sample(self.param['replay_buffer_size'],1)[0],
                              replay_buffer_min_perc=random.sample(self.param['replay_buffer_min_perc'],1)[0],
                              target_update=random.sample(self.param['target_update'],1)[0],
                              policy_update=random.sample(self.param['policy_update'],1)[0],
                              lambda_entropy=random.sample(self.param['lambda_entropy'],1)[0],
                              beta = random.sample(self.param['beta'],1)[0],
                              alpha = random.sample(self.param['alpha'],1)[0],
                              decay_total = random.sample(self.param['decay'],1)[0],
                              per = True,
                              curriculum=random.sample(self.param['curriculum'],1)[0],
                              )
            train.train(test_agent=False,peak=False,uniform_loc=True)

            score = train.agents.test_agent(maze_dataset,n_episodes=1000,len_game=15,num_agents=1)
            if score>=self.best_score:
                self.best_score = score
                self.best_training = train 


class reward_tuning:
    def __init__(self,name, n_iter, maze_dataset,model,vision,
                 len_game, gamma, tau, n_frames, batch_size,
                 lr, lr_step_size, lr_gamma, replay_buffer_size,
                 replay_buffer_min_perc, target_update, policy_update,
                 lambda_entropy, alpha, beta, decay_total, **kwargs):
        # -- best score -- #
        self.best_score = 0

        # -- best agent -- #
        self.best_agent = None

        # -- possible reward distribution -- #
        self.reward_dist = {
            'GOAL' : [1.0,2.0],
            'SEE_GOAL' : [0.0,1.0],
            'DONT_SEE_GOAL': [0.0,1.0],
            'NEW_PLACE': [0.0,1.0],
            'OLD_PLACE': [0.0,1.0],
            'GET_CLOSER': [0.0,1.0],
            'GET_FARTHER' : [0.0,1.0],
            'DIST': [0.0,1.0],
            'DO_ACTION': [0.0,1.0],
            'WALL': [0.0,1.0]
        }

        # -- check if any of the parameters are fixed -- #
        for k in kwargs:
            kwargs[k] = [self.reward_dist[k]]

        self.__tune__(name, n_iter, maze_dataset,model,vision,
                 len_game, gamma, tau, n_frames, batch_size,
                 lr, lr_step_size, lr_gamma, replay_buffer_size,
                 replay_buffer_min_perc, target_update, policy_update,
                 lambda_entropy, alpha, beta, decay_total)
        self.best_agent.save()
        self.best_agent.results()


    def __tune__(self,name, n_iter, maze_dataset,model,vision,
                 len_game, gamma, tau, n_frames, batch_size,
                 lr, lr_step_size, lr_gamma, replay_buffer_size,
                 replay_buffer_min_perc, target_update, policy_update,
                 lambda_entropy, alpha, beta, decay_total):
        
        for i in range(n_iter):
            
            rewards = reward_dist(GOAL = random.uniform(*self.reward_dist['GOAL']),
                                  SEE_GOAL = random.uniform(*self.reward_dist['SEE_GOAL']),
                                  DONT_SEE_GOAL = -1.0*random.uniform(*self.reward_dist['DONT_SEE_GOAL']),
                                  NEW_PLACE = random.uniform(*self.reward_dist['NEW_PLACE']),
                                  OLD_PLACE = -1.0*random.uniform(*self.reward_dist['OLD_PLACE']),
                                  GET_CLOSER = random.uniform(*self.reward_dist['GET_CLOSER']),
                                  GET_FARTHER =-1.0* random.uniform(*self.reward_dist['GET_FARTHER']),
                                  DIST = random.uniform(*self.reward_dist['DIST'])
                                  )

            maze_agent = basic.maze_agents(model,
                               vision=vision,
                               action_type='cardinal',
                               rewards_dist=rewards)
            
            train = basic.Maze_Training(name = name,
                              maze_dataset = maze_dataset,
                              maze_agent = maze_agent,
                              len_game=len_game,
                              n_agents=1,
                              gamma = gamma,
                              tau = tau,
                               batch_size = batch_size,
                              n_frames = n_frames,
                              lr = lr,
                              lr_step_size=lr_step_size,
                              lr_gamma = lr_gamma,
                              replay_buffer_size=replay_buffer_size,
                              replay_buffer_min_perc=replay_buffer_min_perc,
                              target_update=target_update,
                              policy_update=policy_update,
                              lambda_entropy=lambda_entropy,
                              beta = beta,
                              alpha = alpha,
                              decay_total = decay_total,
                              per = True
                              )
            train.train(test_agent=False,peak=False,uniform_loc=True)

            score = train.agents.test_agent(maze_dataset,n_episodes=1000,len_game=15,num_agents=1)
            if score>=self.best_score:
                self.best_score = score
                self.best_agent = train
