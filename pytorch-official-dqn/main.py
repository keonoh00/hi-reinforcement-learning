# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import gymnasium as gym

import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim


from replay_memory import ReplayMemory
from model import DQNModel
from utils import select_action, plot_durations


def train(
    policy_model,
    target_model,
    memory,
    batch_size,
    Transition,
    device,
    gamma,
    optimizer,
):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_model(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[
            0
        ]

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    plt.ion()  # interactive mode

    """
    Hyperparameters and utilities
    """
    device = torch.device("cpu")
    num_episodes = 1000
    batch_size = 256
    gamma = 0.99
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 1000
    target_update = 0.005
    lr = 1e-4
    optimizer = optim.Adam

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_model = DQNModel(
        n_actions=n_actions,
        n_observations=n_observations,
    ).to(device)
    target_model = DQNModel(
        n_actions=n_actions,
        n_observations=n_observations,
    ).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    optimizer = optimizer(policy_model.parameters(), lr=lr)

    """
    Transition:
        A named tuple representing a single transition in our environment.
        It essentially maps (state, action) pairs to their (next_state, reward) result,
        with the state being the screen difference image as described later on.
    """
    Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

    """
    ReplayMemory:
        A cyclic buffer of bounded size that holds the transitions observed recently.
        It also implements a sample() method for selecting a random batch of transitions for training.
    """
    memory = ReplayMemory(10000, Transition=Transition)

    steps_done = 0
    episode_durations = []

    for episode in tqdm(range(num_episodes)):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            steps_done, action = select_action(
                state,
                steps_done,
                epsilon_start,
                epsilon_end,
                epsilon_decay,
                policy_model,
                env,
                device,
            )

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            train(
                policy_model,
                target_model,
                memory,
                batch_size,
                Transition,
                device,
                gamma,
                optimizer,
            )

            target_model_state_dict = target_model.state_dict()
            policy_model_state_dict = policy_model.state_dict()
            for key in policy_model_state_dict:
                target_model_state_dict[key] = (
                    target_update * policy_model_state_dict[key]
                    + (1 - target_update) * target_model_state_dict[key]
                )

            target_model.load_state_dict(target_model_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    print("Complete")
    env.close()
    plt.ioff()
    plt.show()
