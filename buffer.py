from random import sample


class Buffer:
    def __init__(self, size):
        self.buffer = []
        self.size   = size
    
    def add(self, experience):
        if not Buffer.isValid(experience):
            return
        
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
            
        self.buffer.append(experience)
    
    def isValid(experience):
        if experience[3] is "terminated":
            return True
        return False
    
    def sample(self, n):
        # if there are enough experiences
        return sample(self.buffer, n)
        
    def get(self):
        return self.buffer