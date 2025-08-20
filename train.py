import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np
from models import Actor, Critic, save
from buffer import Buffer
from normalizer import StateNormalizer
from copy import deepcopy
import itertools



def Initialize(Env, lr):
    env = gym.make(Env)
    sDim, aDim = env.observation_space.shape[0], env.action_space.shape[0] # type: ignore
    a, c1, c2 = Actor(sDim, aDim), Critic(sDim+aDim), Critic(sDim+aDim),
    optimAct = torch.optim.Adam(a.parameters(), lr=lr)
    optimQ = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=lr)
    log_temp = torch.zeros(1, requires_grad=True)
    optimTemp = torch.optim.Adam([log_temp], lr=lr)
    return a, c1, c2, log_temp, optimAct, optimQ, optimTemp, Buffer(sDim=sDim, aDim=aDim, size=1000000), StateNormalizer(sDim), env # type: ignore
    

def sample(mean:torch.Tensor, log_std:torch.Tensor, with_entropy:bool, scale=0.4):
    std = log_std.exp()
    epsilon = torch.randn_like(mean)
    u = mean + std * epsilon
    tanh_u = torch.tanh(u)
    a = scale * tanh_u
    logP = torch.zeros(1)
    if with_entropy:
        logP_u = -0.5 * ((epsilon**2) + 2*log_std + np.log(2*np.pi))
        logP_u = logP_u.sum(dim=-1, keepdim=True)
        correction = torch.log(1 - tanh_u.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        act_dim = mean.shape[-1]
        logP = logP_u - correction - act_dim * np.log(scale)
    return a, logP


def saveable(step, save=100000):
    return step % save == 0


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
def evaluate(env, norm, actor, episodes=10):
    total_rewards = []
    total_durations = []
    for episode in range(episodes):
        (state, _), truncated, terminal = env.reset(), False, False
        rewards = steps = 0
        while not (terminal or truncated):
            state = norm.prep(state)
            
            with torch.no_grad():
                action, _ = actor.forward(state)
            newState, reward, terminal, truncated, _ = env.step(action.detach().numpy())
            state = newState
            rewards += reward
            steps += 1
        total_rewards.append(rewards)
        total_durations.append(steps)
    return np.average(total_rewards), np.average(total_durations)


def learn(Env="Humanoid-v5", steps=1000000, lr=3e-4, entropy_target=-17, batchSize=256, numOfUpdates=1, target_smoothing=0.005, discount=0.99):
    actor, critic1, critic2, log_temp, optimAct, optimQ, optimTemp, buffer, norm, env = Initialize(Env, lr)
    target1, target2 = deepcopy(critic1), deepcopy(critic2)     # target networks
    temperature = log_temp.exp()
    Avg_Rewards = list()
    
    step = 0
    while step < steps:
        (state, _) = env.reset()
        terminal, truncated = False, False         # Initialize/Reset Enviornment
        
        while not (truncated or terminal):
            norm_state = norm.prep(state)
            
            with torch.no_grad():
                mean, log_std = actor.forward(norm_state)      # type: ignore # select Action a for State s w/ Actor model
            (action, _) = sample(mean, log_std, False)                              # take action in the enviornment

            newState, reward, terminal, truncated, _ = env.step(action.detach().numpy())
            buffer.add(torch.from_numpy(state).detach(), action.detach(), reward, torch.from_numpy(newState).detach(), terminal) # store replay in buffer
            norm.update(state) # update state normalizer
            step += 1  # 1 enviornment step
            
            state = newState
            
            # measure progress
            if step % 10000 == 0:
                avg_reward, avg_duration = evaluate(env, norm, actor)
                Avg_Rewards.append(avg_reward)
                print(f"episode: {step//1000}k; AVG Reward: {Avg_Rewards[-1]:.0f}, AVG Duration: {int(avg_duration)}")
            
            if saveable(step):
                save(
                    env=Env,
                    step=step//1000,
                    actor=actor,
                    critic1=critic1,
                    target1=target1,
                    critic2=critic2,
                    target2=target2
                )
                np.save(f"checkpoints/{Env}/rewards", Avg_Rewards, True)
                torch.save({"mean": norm.mean,
                        "var": norm.var,
                        "epsilon": norm.epsilon}, f"checkpoints/{Env}/norm.pth")
            
            if updatable(buffer.ptr):
                # randomly sample buffer
                batch = buffer.sample(batchSize)
                
                for _ in range(numOfUpdates):
                    states, actions, rewards, newStates, dones = batch     # transpose batch to unpack values
                    states, newStates = norm.prep(states), norm.prep(newStates)
                    
                    # compute targets for Q functions
                    with torch.no_grad():
                        mean, log_std = actor.forward(newStates)
                        newActs, logP = sample(mean, log_std, True)
                        q1, q2 = target1.forward(newStates, newActs), target2.forward(newStates, newActs)
                        q = torch.min(q1, q2) - temperature * logP
                        y = rewards + discount * (1.0 - dones.float()) * q

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
                    
                    # update temperature
                    temp_loss = -(log_temp * (logP.detach() + entropy_target)).mean()
                    optimTemp.zero_grad(set_to_none=True)
                    temp_loss.backward()
                    optimTemp.step()
                    temperature = log_temp.exp()
                    
                    # unfreeze critics
                    for p in itertools.chain(critic1.parameters(), critic2.parameters()):
                        p.requires_grad_(True)
                    
                    # update target network
                    soft_update(critic1, target1, target_smoothing)
                    soft_update(critic2, target2, target_smoothing)
            
    # Save trained models
    save(
        env=Env,
        actor=actor,
        critic1=critic1,
        target1=target1,
        critic2=critic2,
        target2=target2
    )


if __name__ == "__main__":
    learn()