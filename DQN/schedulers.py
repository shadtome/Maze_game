import numpy as np


class curriculumScheduler:
    def __init__(self,start_dist = 1, threshold = 0.8):
        self.threshold = 0.8
        self.start_dist = start_dist

    def check_threshold(self,success_rate):
        return success_rate>self.threshold

    def step(self,n=1):

        self.start_dist +=n
        print('Increasing Difficulty')
            
    
class epsilonDecayScheduler:
    def __init__(self,start_epsilon = 1, end_epsilon = 0.1, decay_total=10000,
                decayType = 'exponential',
                 threshold = 0.70, n_levels=1,
                 mu = 1.0, alpha = 1.0):
        # -- start/end and type of epsilon decays
        self.decayType = decayType
        self.epsilon = [start_epsilon]
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_total = decay_total

        # -- threshold -- #
        self.threshold = threshold

        # -- decay rate based on the 
        self.decay_rate = 0
        self.mu = mu
        self.alpha = alpha

        self.n_levels = n_levels
        self.cur_level = 1

        self.timer_threshold = decay_total/n_levels * mu
        
        self.timer = [0]

        self.reset(start_epsilon=start_epsilon,end_epsilon=end_epsilon,
                   decay_total=decay_total)

    def __decay__(self,t):
        decay = self.start_epsilon * np.power(0.99,self.timer[t]*self.decay_rate)
        self.epsilon[t] = max(self.end_epsilon, decay)

    def step(self):
        upgrades = {'level': False, 'dist':False}
        # -- go through each decay level and increment their timer -- #
        for t in range(self.cur_level):
            self.timer[t]+=1
            self.__decay__(t)
        
        # -- check if the latest epsilon decay level goes down a sufficient level -- #

        if self.timer[-1] >= self.timer_threshold and self.cur_level<self.n_levels:
            self.timer.append(0)
            self.epsilon.append(self.start_epsilon)
            self.cur_level +=1
            print(f' Increasing Level to {self.cur_level}')
            upgrades['level'] = True

        # Check if we need to start the next start distance before decaying
        if self.timer[-1]>= self.timer_threshold*self.alpha and self.cur_level < self.n_levels:
            upgrades['dist']=True
        
        return upgrades

    def check_threshold(self,success_rate):
        return success_rate < self.threshold and self.epsilon[-1] == self.end_epsilon
            
    def reset(self,**kwargs):
        if 'start_epsilon' in kwargs:
            self.start_epsilon = kwargs['start_epsilon']
        if 'end_epsilon' in kwargs:
            self.end_epsilon = kwargs['end_epsilon']
        if 'decay_total' in kwargs:
            self.decay_total = kwargs['decay_total']

        # -- set up the decay rate corresponding to the type of decay -- #
        if self.decayType == 'exponential':
            self.decay_rate = np.log(self.end_epsilon)/(self.decay_total * np.log(0.99))
        
        for t in range(self.cur_level):
            self.timer[t] = t/self.n_levels
            self.__decay__(t) 

    def copy(self):
        return epsilonDecayScheduler(start_epsilon=self.start_epsilon,
                                     end_epsilon=self.end_epsilon,
                                     decay_total=self.decay_total,
                                     decayType=self.decayType,
                                     threshold=self.threshold)