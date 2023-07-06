import torch.nn as nn
from torch.nn import functional as F


class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, observations):
        output = F.relu(self.fc1(observations))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output
