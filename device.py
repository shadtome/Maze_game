import torch

#Global Variable
DEVICE = torch.device('cpu')

def set_device(type_device = 'cpu'):
    """Set the device type for the tensors and models:
    set it as 'cpu' if you want it on the cpu, otherwise, you can 
    set it as 'cuda:0' or any other number if you have more then one
    GPU."""
    global DEVICE 
    DEVICE =  torch.device(type_device)



RANDOM_STATE = 49