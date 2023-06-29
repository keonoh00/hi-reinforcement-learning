import random
import math
from matplotlib import pyplot as plt
from IPython import display

import torch


def select_action(
    env,
    policy_net,
    device,
    state,
    step,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
):
    sample = random.random()
    epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * math.exp(
        -1 * step / epsilon_decay
    )

    step += 1

    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), step
    else:
        return (
            torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.long
            ),
            step,
        )


def plot_durations(episode_durations, show_results=False, is_ipython=False):
    plt.figure(1)
    duration_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_results:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Episode")
    plt.ylabel("Duration")

    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

    if is_ipython:
        if not show_results:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
