from torch import nn


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, 50)
        self.fc2 = nn.Linear(50, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 50)
        self.fc8 = nn.Linear(50, n_actions)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        x = nn.functional.relu(self.fc6(x))
        x = nn.functional.relu(self.fc7(x))
        x = self.fc8(x)
        return x
