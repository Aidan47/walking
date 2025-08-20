import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def ortho_(l: nn.Linear, gain: float):
    torch.nn.init.orthogonal_(l.weight, gain)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, input:int, output:int):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(input, 256, dtype=torch.float32)
        self.l2 = nn.Linear(256, 256, dtype=torch.float32)
        self.mu = nn.Linear(256, output, dtype=torch.float32)
        self.log_std = nn.Linear(256, output, dtype=torch.float32)
        
        # orthogonalize layers
        ortho_(self.l1, np.sqrt(2))
        ortho_(self.l2, np.sqrt(2))
        ortho_(self.mu, 0.01)
        nn.init.constant_(self.log_std.weight, 0.0)
        nn.init.constant_(self.log_std.weight, -0.5)        
    
    def forward(self, s:torch.Tensor):
        x = F.silu(self.l1(s))
        x = F.silu(self.l2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mu, log_std
        

class Critic(nn.Module):
    def __init__(self, input:int):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(input, 256, dtype=torch.float32)
        self.l2 = nn.Linear(256, 256, dtype=torch.float32)
        self.l3 = nn.Linear(256, 1, dtype=torch.float32)
        self.norm = nn.LayerNorm(256)
        
        # orthogonalize layers
        ortho_(self.l1, np.sqrt(2))
        ortho_(self.l2, np.sqrt(2))
        ortho_(self.l3, 0.01)
        
    def forward(self, s:torch.Tensor, a:torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        x = F.silu(self.norm(self.l1(x)))
        x = F.silu(self.norm(self.l2(x)))
        return self.l3(x)
    
    
def save(**kwargs):
    env = kwargs.pop("env", None)
    step = kwargs.pop("step", None)
    
    # save each model
    if step != None:
        for key, value in kwargs.items():
            torch.save(value.state_dict(), f"checkpoints/{env}/{key}_{step}k.pth")
    else:
        for key, value in kwargs.items():
            torch.save(value.state_dict(), f"checkpoints/{env}/{key}.pth")