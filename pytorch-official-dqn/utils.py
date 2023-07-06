import random
import math
import matplotlib.pyplot as plt
import pdb

import torch


def select_action(
    state,
    steps_done,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    policy_model,
    env,
    device,
):
    sample = random.random()
    epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * math.exp(
        -1.0 * steps_done / epsilon_decay
    )
    steps_done += 1
    if sample > epsilon_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return steps_done, policy_model(state).max(1)[1].view(1, 1)

    else:
        return steps_done, torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def plot_durations(episode_durations, show_result=False, fig_idx=0):
    plt.figure(fig_idx)
    durations_tensor = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title = "Result"
    else:
        plt.clf()
        plt.title = "Training..."
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_tensor.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_tensor) >= 100:
        means = durations_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
