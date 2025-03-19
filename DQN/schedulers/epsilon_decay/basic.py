import numpy as np
import os
import json


class BaseEpsilonScheduler:
    def __init__(self,start_epsilon = 1, end_epsilon = 0.1, decay_total=10000,
                decayType = 'exponential'):
        
        # -- start/end and type of epsilon decays
        self.decayType = decayType
        self.epsilon = None
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_total = decay_total
        
        self.timer = None

        self.reset(start_epsilon=start_epsilon,end_epsilon=end_epsilon,
                   decay_total=decay_total)

    def __decay__(self,t):
        decay = self.start_epsilon * np.power(0.99,t*self.decay_rate)
        return max(self.end_epsilon, decay)

    def __setup_rates__(self):
        if self.decayType == 'exponential':
            self.decay_rate = np.log(self.end_epsilon)/(self.decay_total * np.log(0.99))

    def step(self):
        self.timer +=1
        self.epsilon = self.__decay__(self.timer)
    
    def total_time(self):
        return self.decay_total
    
    def __set_timer__(self):
        self.timer = 0

    def __set_epsilon__(self):
        self.epsilon = self.start_epsilon

            
    def reset(self,**kwargs):
        if 'start_epsilon' in kwargs:
            self.start_epsilon = kwargs['start_epsilon']
        if 'end_epsilon' in kwargs:
            self.end_epsilon = kwargs['end_epsilon']
        if 'decay_total' in kwargs:
            self.decay_total = kwargs['decay_total']

        # -- set up the decay rate corresponding to the type of decay -- #
        self.__setup_rates__()
        self.__set_timer__()
        self.__set_epsilon__()


    def copy(self,):
        return self.__class__(**self.__dict__)
    
    def __str__(self):
        s = '----------------------------------\n'
        s += f'Basic epsilon decay scheduler:\n'
        s += f'Start epsilon: {self.start_epsilon}\n'
        s += f'End epsilon: {self.end_epsilon}\n'
        s += f'Decay total: {self.decay_total}\n'
        s += f'Decay rate: {self.decay_rate}\n'
        return s
    
    def __getModelParam__(self):
        return {
            'start_epsilon': self.start_epsilon,
            'end_epsilon': self.end_epsilon,
            'decay_total': self.decay_total
        }
    
    def save(self,filedir):
        
        param = self.__getModelParam__()

        with open(os.path.join(filedir,'EpsilonDecayParameters.json'),'w') as f:
            json.dump(param,f,indent=4)
