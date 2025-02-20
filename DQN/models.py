
import torch
import torch.nn as nn
import torchvision.models as models
import Maze_env


class CNN_Basic(nn.Module):
    def __init__(self,state_shape,n_actions):
        """THe Convolutional Neural network
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__()
        self.state_shape = state_shape # Shape of the images for the local information
        self.n_actions = n_actions # number of actions
        h = state_shape[1]
        w = state_shape[2]

        self.name = 'CNN_Basic'
       
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
    
    
    def copy(self):
        q_copy = CNN_Basic(self.state_shape,self.n_actions)
        q_copy.load_state_dict(self.state_dict())
        return q_copy
    
class basic_NN(nn.Module):
    def __init__(self,state_shape,n_actions):
        """Deep Minds architecture.
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__()
        self.state_shape = state_shape # Shape of the images for the local information
        self.n_actions = n_actions # number of actions
        h = state_shape[1]
        w = state_shape[2]
        input = h*w*3
        self.name = 'basic_NN'
        
        self.functions = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128,n_actions)
        )


    def forward(self,x,y):
        
        return self.functions(x)
    
    
    def copy(self):
        q_copy = basic_NN(self.state_shape,self.n_actions)
        q_copy.load_state_dict(self.state_dict())
        return q_copy

class CNN_version1(nn.Module):
    def __init__(self,state_shape,n_actions):
        """THe Convolutional Neural network
            State_shape: The state shape will be of the form (3,h,w) where h is the height and
                        w the width of the image.  The images are neighborhood observations of the agent
            n_actions: the number of actions the agent can take: traditional will be 5"""
        super().__init__()
        self.state_shape = state_shape # Shape of the images for the local information
        self.n_actions = n_actions # number of actions
        h = state_shape[1]
        w = state_shape[2]

        self.name = 'CNN_version1'


       
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

    def forward(self,x,y):
        
        x = self.CNN_function(x)
        
        combined = torch.cat((x,y),dim=-1)
        result = self.final_function(combined)
        return result
    
    
    def copy(self):
        q_copy = CNN_version1(self.state_shape,self.n_actions)
        q_copy.load_state_dict(self.state_dict())
        return q_copy
    
    

metadata = {
    'CNN_Basic' : CNN_Basic,
    'basic_NN': basic_NN,
    'CNN_version1': CNN_version1,
}