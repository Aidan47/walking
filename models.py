import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Class (Actor)
    in:     State
    out:    Mean & Std Dev (of Gaussian Distribution)
* If testing: action = mean *
* If training: action = random selection from dist. *

Class (Critic)
    in:     State & Action
    out:    Reward
* Trained on replay buffer of Actor *
'''


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(376, 32, dtype=torch.float64)
        self.l2 = nn.Linear(32, 64, dtype=torch.float64)
        self.l3 = nn.Linear(64, 32, dtype=torch.float64)
        self.l4 = nn.Linear(32, 16, dtype=torch.float64)
        self.l5 = nn.Linear(16, 2, dtype=torch.float64)
    
    def forward(self, x:torch.Tensor):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
    
def expand(self):
    return self[0], self[1]
        

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(365, 256, dtype=torch.float64)
        self.l2 = nn.Linear(256, 128, dtype=torch.float64)
        self.l3 = nn.Linear(128, 128, dtype=torch.float64)
        self.l4 = nn.Linear(128, 64, dtype=torch.float64)
        self.l5 = nn.Linear(64, 32, dtype=torch.float64)
        self.l6 = nn.Linear(32, 1, dtype=torch.float64)
        
    def forward(self, x:torch.Tensor):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)