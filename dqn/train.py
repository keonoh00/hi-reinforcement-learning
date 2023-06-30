import numpy as np
from tqdm import tqdm

import torch

from agent import DQNAgent
from env import BlobEnv

if __name__ == "__main__":
    episodes = 20_000
    learning_rate = 0.001

    env = BlobEnv(area_size=10)

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr=learning_rate,
    )

    for episode in tqdm(range(episodes), ascii=True, unit="episode"):
        writer = agent.writer

        episode_reward = 0
        step = 1

        state = env.reset()
        done = False

        while not done:
            if np.random.random() > agent.epsilon:
                action = np.argmax(agent.get_q_values(state))
            else:
                action = np.random.randint(0, env.action_space)

            next_state, reward, done = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((state, action, reward, next_state, done))
            agent.train(done, step=step)

            state = next_state
            step += 1
