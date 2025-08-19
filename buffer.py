import torch
import numpy as np



class Buffer:
    def __init__(self, size, sDim, aDim, dtype=torch.float32):
        self.size   = size
        
        self.S      = torch.zeros((size, sDim), dtype=dtype)
        self.A      = torch.zeros((size, aDim), dtype=dtype)
        self.R      = torch.zeros((size, 1), dtype=dtype)
        self.S2     = torch.zeros((size, sDim), dtype=dtype)
        self.done   = torch.zeros((size, 1), dtype=dtype)
        
        self.ptr = 0
        self.full = False
    
    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.S[i] = s
        self.A[i] = a
        self.R[i, 0] = r
        self.S2[i] = s2
        self.done[i, 0] = d
        self.ptr = (i + 1) % self.size
        if self.ptr == 0:
            self.full = True
            
    def __len__(self):
        return self.size if self.full is True else self.ptr
    
    def sample(self, batch_size):
        n = len(self)
        idx = np.random.randint(0, n, (batch_size,))
        return (self.S[idx],
                self.A[idx],
                self.R[idx],
                self.S2[idx],
                self.done[idx])