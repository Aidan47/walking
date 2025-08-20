import numpy as np


class StateNormalizer:
    def __init__(self, state_dim, epsilon=1e-8):
        self.count = 0
        self.mean = np.zeros(state_dim, dtype=np.float32)
        self.var  = np.ones(state_dim, dtype=np.float32)
        self.epsilon = epsilon
        
    def update(self, s):    # only for individual states
        self.count += 1
        delta = s - self.mean
        self.mean += delta / self.count
        self.var += delta * (s - self.mean) / self.count
    
    def normalize(self, s):
        return (s - self.mean) / (np.sqrt(self.var) + self.epsilon)
    
    def clip(self, s, min, max):
        return np.clip(s, min, max)
    
    def prep(self, s):
        s = self.normalize(s)
        return self.clip(s, -5, 5)