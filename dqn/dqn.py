from torch import nn


class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.model = nn.Sequential(
            nn.Conv2d(observation_space, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1920, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

    def forward(self, x):
        return self.model(x)
