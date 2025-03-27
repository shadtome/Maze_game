
import torch
import torch.nn as nn
import torchvision.models as models
import Maze_env


class AgentNN(nn.Module):
    def __init__(self,name,state_shape,n_actions, **kwargs):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.name = name

    def __getParams__(self):
        params = {
            'state_shape':self.state_shape,
            'n_actions':self.n_actions
        }
        return params

    def copy(self):
        q_copy =self.__class__(**self.__getParams__())
        q_copy.load_state_dict(self.state_dict())
        return q_copy


class CNN_Basic(AgentNN):
    def __init__(self,state_shape,n_actions):
        """THe Convolutional Neural network
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__('CNN_Basic',state_shape,n_actions)
        h = state_shape[1]
        w = state_shape[2]
       
        self.CNN_function= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=12,out_channels=24,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24*int((h-3)/2)*int((w-3)/2),n_actions)
        )
        

    def forward(self,x,y):
        
        x = self.CNN_function(x)
        
        return x
    
    
class basic_NN(AgentNN):
    def __init__(self,state_shape,n_actions):
        """Deep Minds architecture.
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__('basic_NN',state_shape,n_actions)
        h = state_shape[1]
        w = state_shape[2]
        input = h*w*3
        
        self.functions = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128,n_actions)
        )


    def forward(self,x,y):
        
        return self.functions(x)
    

class CNN_version1(AgentNN):
    def __init__(self,state_shape,n_actions):
        """THe Convolutional Neural network
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__('CNN_version1',state_shape,n_actions)
        h = state_shape[1]
        w = state_shape[2]
       
        self.CNN_function= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*int((h-3)/2)*int((w-3)/2),32),
            nn.ReLU(),
        )

        self.final_function =nn.Sequential(
            nn.Linear(32 + 4,32),
            nn.ReLU(),
            nn.Linear(32,12),
            nn.ReLU(),
            nn.Linear(12,self.n_actions)
        ) 

    def freeze_base(self,freeze=False):
        for param in self.CNN_function.parameters():
            param.requires_grad=freeze


    def forward(self,x,y):
        
        x = self.CNN_function(x)
        
        combined = torch.cat((x,y),dim=-1)
        result = self.final_function(combined)
        return result
    
    
class CNN_version2(AgentNN):
    def __init__(self,state_shape,n_actions):
        """THe Convolutional Neural network
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__('CNN_version2',state_shape,n_actions)
        h = state_shape[1]
        w = state_shape[2]


       
        self.CNN_function= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(64*int((h-3)/2)*int((w-3)/2),32),
            nn.ReLU(),
        )

        self.final_function =nn.Sequential(
            nn.Linear(32 + 4,32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32,12),
            nn.ReLU(),
            nn.Linear(12,self.n_actions)
        ) 

    def forward(self,x,y):
        
        x = self.CNN_function(x)
        
        combined = torch.cat((x,y),dim=-1)
        result = self.final_function(combined)
        return result
    

    
class MultiHead(AgentNN):
    def __init__(self,state_shape,n_actions,current_game = None):
        """MultiHead Agent Based on its Task"""
        super().__init__('MultiHead',state_shape,n_actions)
        self.current_game = current_game
        self.base= None
        self.heads = {}

    def set_game(self,game):
        """Set the current mode for the agent"""
        if game not in self.heads:
            raise ValueError(f'Invalid mode {game}')
        self.current_game = game

    def forward(self,x,y):

        x = self.base(x)
        combined = torch.cat((x,y),dim=-1)
        result = self.heads[self.current_game](combined)
        return result
    
    def freeze_base(self,freeze=False):
        for param in self.base.parameters():
            param.requires_grad=freeze
        
    
    def __getParams__(self):
        params = super().__getParams__()
        params['current_game'] = self.current_game
        return params
    
class MH_CNN(MultiHead):
    def __init__(self, state_shape, n_actions,current_game=None):
        h = state_shape[1]
        w = state_shape[2]
        super().__init__(state_shape, n_actions,current_game)

        self.base= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*int((h-3)/2)*int((w-3)/2),32),
            nn.ReLU(),
        )
        
        goal_head = nn.Sequential(
            nn.Linear(32 + 4,32),
            nn.ReLU(),
            nn.Linear(32,12),
            nn.ReLU(),
            nn.Linear(12,n_actions)
        )
        
        hunger_games_head = nn.Sequential(
            nn.Linear(32 + 4,32),
            nn.ReLU(),
            nn.Linear(32,12),
            nn.ReLU(),
            nn.Linear(12,n_actions)
        )

        self.heads['Maze Runner'] = goal_head
        self.heads['Hunger Games'] = hunger_games_head

metadata = {
    'CNN_Basic' : CNN_Basic,
    'basic_NN': basic_NN,
    'CNN_version1': CNN_version1,
    'CNN_version2': CNN_version2,
    'MultiHead': MultiHead,
    'MH_CNN': MH_CNN
}