from torch.optim.lr_scheduler import _LRScheduler
import os
import json


class BasicHeadLR(_LRScheduler):
    def __init__(self,optimizer,step_size,gamma, last_epoch = -1):
        self.step_size = step_size
        self.gamma = gamma
        super(BasicHeadLR,self).__init__(optimizer,last_epoch)
        
        
    def get_lr(self):
        groups = []
        if self.last_epoch % self.step_size == 0:
            for group in self.optimizer.param_groups:
                if any(p.grad is not None for p in group['params']):
                    groups.append(group['lr']*self.gamma)
                else:
                    groups.append(group['lr'])
        return groups


    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        super(BasicHeadLR, self).step()

    def __str__(self):
        s = '--------------------------------------\n'
        s += f'Basis learning rate scheduler:\n'
        s += f'Step size: {self.step_size}\n'
        s += f'Gamma: {self.gamma}'
        return s
    
    def __getModelParams__(self):
        return {'step_size':self.step_size,'gamma':self.gamma}

    def save(self,filedir):
        params = self.__getModelParams__()
        with open(os.path.join(filedir,'LearningRateParameters.json'),'w') as f:
            json.dump(params,f)
