
from torch.optim.lr_scheduler import _LRScheduler


class VariableLR(_LRScheduler):
    def __init__(self,optimizer,step_size,head_step_size,gamma,head_gamma, last_epoch = -1):
        self.step_size = step_size
        self.head_step_size = head_step_size
        self.gamma = gamma
        self.head_gamma = head_gamma
        super(VariableLR,self).__init__(optimizer,last_epoch)
        
        
    def get_lr(self):
        groups = []
        if self.last_epoch % self.step_size == 0 or self.last_epoch % self.head_step_size==0:
            for i,group in enumerate(self.optimizer.param_groups):
                if i in [0,1] and self.last_epoch % self.step_size == 0:
                    if any(p.grad is not None for p in group['params']):
                        groups.append(group['lr']*self.gamma)
                elif i>1 and self.last_epoch % self.head_step_size ==0:
                    if any(p.grad is not None for p in group['params']):
                        groups.append(group['lr']*self.head_gamma)
                else:
                    groups.append(group['lr'])
        return groups


    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        super(VariableLR, self).step()

    def __str__(self):
        s = '--------------------------------------\n'
        s += f'Variable learning rate scheduler:\n'
        s += f'Step size: {self.step_size}\n'
        s += f'Head Step size: {self.head_step_size}\n'
        s += f'Gamma: {self.gamma}\n'
        s += f'Head Gamma: {self.head_gamma}'
        return s