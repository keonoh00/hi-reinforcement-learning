import time
import numpy as np
import random

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from memory import ReplayMemory


class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.model = nn.Sequential(
            nn.Linear(self.observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space),
        )

    def forward(self, x):
        return self.model(x)
