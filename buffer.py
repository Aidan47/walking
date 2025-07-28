import torch
from random import sample


class Buffer:
    def __init__(self, size):
        self.buffer = []
        self.size   = size
    
    def add(self, experience):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        # convert all ndarrays to tensors
        experience = [torch.from_numpy(ele) for i, ele in enumerate(experience) if i != 2]
        self.buffer.append(experience)
    
    def sample(self, n):
        # if there are enough experiences
        return sample(self.buffer, n)
        
    def get(self):
        return self.buffer