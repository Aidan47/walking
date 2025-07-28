import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np
from models import Actor, Critic, expand
from buffer import Buffer



def Initialize():
    env = gym.make("Humanoid-v4")
    return Actor(), Critic(), Critic(), Buffer(size=1000000), env
    

def updatable(step, start=10000):
    if step > start:
        return True
    return False


def update(actor, critic1, critic2, buffer, n, batchSize):
    for _ in range(n):
        # randomly sample buffer
        batch = buffer.sample(batchSize)
        
        '''
        compute targets for Q functions
        update q functions (gradient descent)
        update policy (gradient ascent)
        update target network        
        '''
    pass


def learn(steps=3000000, batchSize=256, numOfUpdates=1):
    actor, critic1, critic2, buffer, env = Initialize()
    target1, target2 = critic1, critic2     # target networks
    
    for step in range(steps):
        (state, _), truncated, terminal = env.reset(), False, False         # Initialize/Reset Enviornment
        
        while not terminal and not truncated:
            mean, std = expand(actor.forward(torch.from_numpy(state)))      # type: ignore # select Action a for State s w/ Actor model
            action = torch.normal(mean, std)                                # take action in the enviornment
            
            state, reward, newState, truncated, terminal = env.step(action.detach().numpy())
            
            if not terminal:
                buffer.add((state, action, reward, newState))   # store replay in buffer
                state = newState
                
            if updatable(step):
                update(actor, critic1, critic2, buffer, numOfUpdates, batchSize)
            
    # Save trained models
    torch.save(actor.state_dict(), 'actor_weights.pth')
    torch.save(target1.state_dict(), 'critic_weights.pth')
    return


if __name__ == "__main__":
    learn()