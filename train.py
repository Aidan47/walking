import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np
from models import Actor, Critic, save
from buffer import Buffer
from copy import deepcopy
import itertools



def Initialize(lr):
    env = gym.make("Humanoid-v4")
    a, c1, c2 = Actor(), Critic(), Critic(),
    optimAct = torch.optim.Adam(a.parameters(), lr=lr)
    optimQ = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=lr)
    return a, c1, c2, optimAct, optimQ, Buffer(sDim=env.observation_space.shape[0], aDim=env.action_space.shape[0], size=1000000), env # type: ignore
    

def sample(mean:torch.Tensor, log_std:torch.Tensor, with_entropy:bool, scale=0.4):
    std = log_std.exp()
    epsilon = torch.randn_like(mean)
    u = mean + std * epsilon
    tanh_u = torch.tanh(u)
    a = scale * tanh_u
    logP = 0
    if with_entropy:
        logP_u = -0.5 * ((epsilon**2) + 2*log_std + np.log(2*np.pi))
        logP_u = logP_u.sum(dim=-1, keepdim=True)
        correction = torch.log(1 - tanh_u.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        act_dim = mean.shape[-1]
        logP = logP_u - correction - act_dim * np.log(scale)
    return a, logP


def updatable(step, start=10000):
    if step > start:
        return True
    return False


def soft_update(Q, T, p):
    for q, t in zip(Q.parameters(), T.parameters()):
        t.data.lerp_(q.data, p)
    for q, t in zip(Q.buffers(), T.buffers()):
        t.data.copy_(q.data)


@torch.no_grad()
def evaluate(env, actor, episodes=3):
    AVG_reward = 0
    for episode in range(episodes):
        (state, _), truncated, terminal = env.reset(), False, False
        while not terminal or truncated:
            with torch.no_grad():
                action, _ = actor.forward(torch.from_numpy(state).float())
            newState, reward, terminal, truncated, _ = env.step(action.detach().numpy())
            state = newState
            AVG_reward += (1/episodes) * reward    # distrubutivity shows <- == (∑ reward_per_episode) / episodes
    return AVG_reward


def learn(steps=10000000, lr=3e-4, reward_scale=20, temperature=0.2, batchSize=256, numOfUpdates=1, target_smoothing=0.005, discount=0.99):
    actor, critic1, critic2, optimAct, optimQ, buffer, env = Initialize(lr)
    target1, target2 = deepcopy(critic1), deepcopy(critic2)                     # target networks
    rewards = np.ndarray([])
    
    step = 0
    while step < steps:
        (state, _), truncated, terminal = env.reset(), False, False             # Initialize/Reset Enviornment

        while not terminal and not truncated:
            with torch.no_grad():
                mean, log_std = actor.forward(torch.from_numpy(state).float())  # type: ignore # select Action a for State s w/ Actor model
            (action, _) = sample(mean, log_std, False)                          # take action in the enviornment

            newState, reward, terminal, truncated, info = env.step(action.detach().numpy())
            step += 1
            done = terminal or truncated
            buffer.add(torch.from_numpy(state).detach(), action.detach(), reward, torch.from_numpy(newState).detach(), done) # store replay in buffer
            state = newState
            
            # measure progress
            if step % 1000 == 0:
                rewards = np.append(rewards, evaluate(env, actor))
                print(f"episode: {step}; AVG Reward: {rewards[-1]:.4f}")
            
            if updatable(buffer.ptr):
                # randomly sample buffer
                batch = buffer.sample(batchSize)
                
                for _ in range(numOfUpdates):
                    states, actions, rewards, newStates, done = batch     # transpose batch to unpack values

                    # compute targets for Q functions
                    with torch.no_grad():
                        mean, log_std = actor.forward(newStates)
                        newActs, logP = sample(mean, log_std, True)
                        q1, q2 = target1.forward(newStates, newActs), target2.forward(newStates, newActs)
                        q = torch.min(q1, q2) - temperature * logP
                        y = rewards + discount * (1.0 - done.float()) * q

                    # update q functions (gradient descent)
                    critic_loss = 0.5 * (F.mse_loss(critic1.forward(states, actions), y) + F.mse_loss(critic2.forward(states, actions), y))
                    optimQ.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    optimQ.step()
                    
                    # freeze critics
                    for p in itertools.chain(critic1.parameters(), critic2.parameters()):
                        p.requires_grad_(False)

                    # update policy (gradient ascent)
                    mean, log_std = actor.forward(states)
                    acts, logP = sample(mean, log_std, True)
                    qMin = torch.min(critic1.forward(states, acts), critic2.forward(states, acts))
                    actor_loss = torch.mean(temperature * logP - qMin)
                    optimAct.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    optimAct.step()
                    
                    # unfreeze critics
                    for p in itertools.chain(critic1.parameters(), critic2.parameters()):
                        p.requires_grad_(True)
                    
                    # update target network
                    soft_update(critic1, target1, target_smoothing)
                    soft_update(critic2, target2, target_smoothing)
            
    # Save trained models
    save(
        actor=actor,
        critic1=critic1,
        target1=target1,
        critic2=critic2,
        target2=target2
    )
    
    # Save rewards
    np.save("rewards", rewards, True)


if __name__ == "__main__":
    learn()