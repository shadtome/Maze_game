import numpy as np
from DQN.schedulers.epsilon_decay.basic import BaseEpsilonScheduler
            
    
class GradientEpsilonScheduler(BaseEpsilonScheduler):
    def __init__(self,start_epsilon = 1, end_epsilon = 0.1, decay_total=10000,
                decayType = 'exponential', n_levels=1,
                 mu = 1, alpha = 1.0):
        """mu is (0,1], alpha is (0,1]"""
        # -- rate at which to start the next level -- #
        self.mu = mu
        # -- mu increase rate -- #
        self.mu_rate = 1
        # -- rate at which to introduce the next level before training -- #
        self.alpha = alpha

        # -- number of levels to epsilon decay -- #
        self.n_levels = n_levels
        self.cur_level = 1

        self.timer_threshold = decay_total/n_levels
        
        super().__init__(start_epsilon,end_epsilon,decay_total,decayType)


        self.reset(start_epsilon=start_epsilon,end_epsilon=end_epsilon,
                   decay_total=decay_total)

    def __mu_rate__(self,l):
        return np.power(0.99,self.mu_rate*l)

    def __inc_mu__(self,l):
        self.mu = self.mu*self.__mu_rate__(l)

    def step(self):
        upgrades = {'level': False, 'dist':False}
        # -- go through each decay level and increment their timer -- #
        for t in range(self.cur_level):
            self.timer[t]+=1
            self.epsilon[t] = self.__decay__(self.timer[t])
        
        # -- check if the latest epsilon decay level goes down a sufficient level -- #
        # Check if we need to start the next start distance before decaying
        if self.timer[-1] == int(self.timer_threshold*self.mu*self.alpha) and self.cur_level < self.n_levels:
            upgrades['dist']=True

        if self.timer[-1] == int(self.timer_threshold*self.mu) and self.cur_level<self.n_levels:
            self.timer.append(0)
            self.epsilon.append(self.start_epsilon)
            self.cur_level +=1
            self.__inc_mu__(self.cur_level)
            print(f' Increasing Level to {self.cur_level}')
            upgrades['level'] = True

        return upgrades
    
    def total_time(self):
        if self.n_levels == 1:
            return self.decay_total
        else:
            if abs(self.mu - 1.0)<0.00001:
                return 2*self.decay_total + self.timer_threshold
            else:
                time = self.timer_threshold
                time = time * self.mu
                time = time * (1 - self.__mu_rate__(self.n_levels+1))
                time = time/(1-self.__mu_rate__(1))
                time = time + self.decay_total + self.timer_threshold
                return int(time)

    def __set_timer__(self):
        self.timer = [0]

    def __set_epsilon__(self):
        self.epsilon = [self.start_epsilon]

    def __setup_rates__(self):
        super().__setup_rates__()
        if self.decayType == 'exponential':
            self.mu_rate = -1*np.log(self.mu)/(self.n_levels*np.log(0.99))
            
    def reset(self,**kwargs):
        super().reset(**kwargs)

    def __str__(self):
        s = super().__str__()
        s += f'starting mu: {self.mu}\n'
        s += f'mu rate: {self.mu_rate}\n'
        s += f'alpha: {self.alpha}\n'
        s += f'timer threshold: {self.timer_threshold}'
        return s