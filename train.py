import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from models import Actor, Critic
from buffer import Buffer



def Initialize():
    return Actor, Critic, Buffer
    

def updatable():
    pass


def update(buffer, n, exs):
    for _ in range(n):
        # randomly sample buffer
        batch = buffer.sample(exs)
        
        '''
        compute targets for Q functions
        update q functions (gradient descent)
        update policy (gradient ascent)
        update target network        
        '''
    pass


def main(converging=True):
    
    model1, model2, buffer = Initialize()
    
    # Initialize Enviornment
    
    while converging:
        # select Action a for State s w/ Actor model
        
        # take action in the enviornment
        
        if not terminal:
            # store old state, action, reward, new state, and new state terminality
            buffer.add("(s, a, r, s', t)")
        else:
            # reset enviornment
            
        if updatable():
            update(numOfUpdates)
            
            
    # Save trained models
    
    return


if __name__ == "__main__":
    pass
