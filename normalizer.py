import torch


class StateNormalizer:
    def __init__(self, state_dim, epsilon=1e-8):
        self.count = 0
        self.mean = torch.zeros(state_dim, dtype=torch.float32, requires_grad=False)
        self.var  = torch.ones(state_dim, dtype=torch.float32, requires_grad=False)
        self.epsilon = epsilon
        
    def update(self, s):    # only for individual states
        self.count += 1
        delta = s - self.mean
        self.mean += delta / self.count
        self.var += delta * (s - self.mean) / self.count
    
    def normalize(self, s):
        return (s - self.mean) / (torch.sqrt(self.var) + self.epsilon)
    
    def clip(self, s, min, max):
        return torch.clamp(s, min, max)