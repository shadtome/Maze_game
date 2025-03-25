

class BasicRewardFun:
    def __init__(self, **kwargs):

        self.rewards = self.__setup__()
        
        self.change_rewards(**kwargs)

    def change_rewards(self, **kwargs):
        """Takes any of the key words for the 
        goals and changes them."""
        for k in kwargs:
            if k in self.rewards:
                self.rewards[k] = kwargs[k]

    def __setup__(self):
        rewards = {
            'GOAL' : 1.0,
            'FAIL' : -1.0
        }
        return rewards

    def __getitem__(self,item):
        return self.rewards[item]
    
    def __setitem__(self,key,value):
        self.rewards[key] = value

    def __str__(self):
        s = '----------------Rewards Distribution --------------\n'
        for k, v in self.rewards.items():
            s += f'{k}: {v}\n'
        s += '-----------------------------------------------------'
        return s