import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Class (Actor)
    in:     State
    out:    Mean & Std Dev (of Gaussian Distribution)
* If testing: action = mean *
* If training: action = random selection from dist. *

Class (Critic)
    in:     State & Action
    out:    Reward
* Trained on replay buffer of Actor *
'''