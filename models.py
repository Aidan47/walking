import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(376, 256, dtype=torch.float32)
        self.l2 = nn.Linear(256, 256, dtype=torch.float32)
        self.mu = nn.Linear(256, 17, dtype=torch.float32)
        self.log_std = nn.Linear(256, 17, dtype=torch.float32)
    
    def forward(self, s:torch.Tensor):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mu, log_std
        

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(393, 256, dtype=torch.float32)
        self.l2 = nn.Linear(256, 256, dtype=torch.float32)
        self.l3 = nn.Linear(256, 1, dtype=torch.float32)
        
    def forward(self, s:torch.Tensor, a:torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
    
    
def save(**kwargs):
    for key, value in kwargs.items():
        torch.save(value.state_dict(), f"checkpoints/{key}.pth")    # save each model