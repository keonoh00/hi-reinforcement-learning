import os
import numpy as np
from tqdm import tqdm

import torch

from agent import DQNAgent
from env import BlobEnv

if __name__ == "__main__":
    episodes = 20_000
    learning_rate = 0.001

    save_every = 1_000

    env = BlobEnv()

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr=learning_rate,
        device=torch.device("cuda:2"),
    )

    if not os.path.exists("./episodes"):
        os.mkdir("./episodes")

    for episode in tqdm(range(episodes), unit="episodes"):
        agent.epsilon = 1 - (episode / episodes)

        frames = []

        writer = agent.writer

        episode_reward = 0
        step = 1

        state = env.reset()
        done = False

        while not done:
            if np.random.random() > agent.epsilon:
                action = np.argmax(agent.get_q_values(state).detach().to("cpu").numpy())
            else:
                action = np.random.randint(0, env.action_space)

            next_state, reward, done = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((state, action, reward, next_state, done))
            agent.train(done)

            agent.writer.add_scalar("Reward", reward, global_step=step)

            state = next_state
            step += 1

            step_image = env.get_image()
            step_image = step_image.resize((300, 300))

            frames.append(step_image)

        agent.writer.add_scalar("Episode Reward", episode_reward, global_step=episode)

        if episode % save_every == 0:
            frame_one = frames[0]
            frame_one.save(
                f"./episodes/{episode}.gif",
                format="GIF",
                append_images=frames,
                save_all=True,
                duration=100,
                loop=0,
            )
