import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from matplotlib import pyplot as plt
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, target):
        state      = np.expand_dims(state, 0)
        self.buffer.append((state, action, target))
    
    def sample(self, batch_size):
        state, action, target = zip(*random.sample(self.buffer, batch_size))
        
        return np.concatenate(state), action, target
    
    def sample_for_GP(self):
        state, action, target = zip(*self.buffer)
        
        return np.concatenate(state), action, target
    def __len__(self):
        return len(self.buffer)