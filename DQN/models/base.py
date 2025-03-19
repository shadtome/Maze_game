
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
    def __init__(self,state_shape,n_actions,n_heads = 1):
        """Mutlihead neural network based on level"""
        super().__init__('MultiHead',state_shape,n_actions)
        self.n_heads = n_heads
        h = state_shape[1]
        w = state_shape[2]


       
        self.base= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*int((h-3)/2)*int((w-3)/2),32),
            nn.ReLU(),
        )

        self.combine_base =nn.Sequential(
            nn.Linear(32 + 4,32),
            nn.ReLU(),
            nn.Linear(32,12),
            nn.ReLU()
        ) 

        self.heads = []
        for i in range(n_heads):
            self.heads.append(nn.Linear(12,n_actions))

    def forward(self,x,y,head):
        if isinstance(head,int):
            head = [head]

        x = self.base(x)
        batch_size = x.size(0)
        combined = torch.cat((x,y),dim=-1)
        result = self.combine_base(combined)
        
        selected_outputs = []
        for i in range(batch_size):
            if head[i]>=self.n_heads:
                selected_head = self.heads[-1]
            else:
                selected_head = self.heads[head[i]]
            selected_output = selected_head(result[i])  
            selected_outputs.append(selected_output)

        return torch.stack(selected_outputs)
    
    def __getParams__(self):
        params = super().__getParams__()
        params['n_heads'] = self.n_heads
        return params
    

metadata = {
    'CNN_Basic' : CNN_Basic,
    'basic_NN': basic_NN,
    'CNN_version1': CNN_version1,
    'CNN_version2': CNN_version2,
    'MultiHead': MultiHead
}