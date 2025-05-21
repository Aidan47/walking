import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from models import Actor, Critic
from buffer import Buffer


'''
Training

for trial
    action = sample(actor.forward(state))
    get new observation
    add expeirence to replay buffer
    
    if condition:
        set = torch (get random dataset from buffer)
        train critic with set
        
    update 

'''