import time
import numpy as np

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from memory import ReplayMemory
from dqn import DQN


class DQNAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        batch_size=64,
        lr=0.001,
        epsilon=1,
        discount=0.99,
        replay_memory_size=50_000,
        min_replay_memory_size=1_000,
        log_dir="./logs",
        model_name="256x256",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.discount = discount
        self.epsilon = epsilon
        self.device = device

        # Tuple of (state, action, reward, next_state, done).
        self.memory = ReplayMemory(capacity=replay_memory_size)

        self.model = DQN(self.observation_space, self.action_space).to(
            self.device,
        )
        self.target_model = DQN(self.observation_space, self.action_space).to(
            self.device,
        )

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.target_loss = nn.MSELoss()
        self.target_optimizer = optim.Adam(self.target_model.parameters(), lr=lr)

        self.update_target()

        self.writer = SummaryWriter(
            log_dir=f"{log_dir}/{model_name}-{int(time.time())}"
        )

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_replay_memory(self, transition):
        """
        Update replay memory with transition.

        Parameters
        ----------
        transition : tuple
            Tuple of (state, action, reward, next_state, done).
        """
        self.memory.push(transition)

    def get_q_values(self, state):
        """
        Get Q values from state.

        Parameters
        ----------
        state : torch.Tensor
            State of the environment.

        Returns
        -------
        torch.Tensor
            Q values from state.
        """
        return self.model(
            np.array(state).reshape(-1, *state.shape) / 255,  # Normalize with 255
        )[0]

    def train(self, terminal_state, step):
        """
        Train the model.

        Parameters
        ----------
        terminal_state : bool
            Whether the state is terminal or not.
        step : int
            Current step.
        """
        if len(self.memory) < self.min_replay_memory_size:
            return

        minibatch = self.memory.sample_batch(self.batch_size)

        current_states = (
            torch.tensor(np.array([transition[0] for transition in minibatch]))
            / 255  # Normalize with 255
        )
        current_states = current_states.to(self.device)

        # save current_states as text file for debugging
        # np.savetxt("current_states.txt", current_states)

        current_qs_list = self.model(current_states)

        new_current_states = (
            torch.tensor(np.array([transition[3] for transition in minibatch]))
            / 255  # Normalize with 255
        )
        new_current_states = new_current_states.to(self.device)
        future_qs_list = self.target_model(new_current_states)

        X = []  # Current states
        y = []  # Future states

        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index].detach().to("cpu").numpy()
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.optimizer.zero_grad()

        X = torch.tensor(np.array(X)) / 255
        X = X.to(self.device)
        y = torch.tensor(np.array(y))
        y = y.to(self.device)

        y_pred = self.model(X)

        loss = self.loss(y_pred, y)
        loss.backward()

        self.optimizer.step()

        self.writer.add_scalar("Loss", loss, global_step=step)

        if terminal_state:
            self.update_target()
