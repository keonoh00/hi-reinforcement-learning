# https://www.youtube.com/watch?v=G92TF4xYQcU
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time

from blob import Blob

"""
Observation Space:
  1. Relative position of the food to the player (x1, y1)
  2. Relative position of the enemy to the player (x2, y2)
"""

AREA_SIZE = 10

NUM_EPISODES = 500000
SHOW_EVERY = 3000
NUM_STEPS_PER_EPISODE = 500

MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.9
EPS_DECAY = 0.9998

start_q_table = None  # or filename in pickle format

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_ID = 1  # player key in dict
FOOD_ID = 2  # food key in dict
ENEMY_ID = 3  # enemy key in ENEMY_ID

color_dict = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}  # color dict (B, G, R)


if __name__ == "__main__":
    style.use("ggplot")

    if start_q_table is None:
        q_table = {}

        for x1 in range(-AREA_SIZE + 1, AREA_SIZE):
            for y1 in range(-AREA_SIZE + 1, AREA_SIZE):
                for x2 in range(-AREA_SIZE + 1, AREA_SIZE):
                    for y2 in range(-AREA_SIZE + 1, AREA_SIZE):
                        # Each observation space needs 4 actions
                        q_table[((x1, y1), (x2, y2))] = [
                            np.random.uniform(-5, 0) for _ in range(4)
                        ]

    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        player = Blob(AREA_SIZE)
        food = Blob(AREA_SIZE)
        enemy = Blob(AREA_SIZE)

        if episode % SHOW_EVERY == 0:
            print(f"on #{episode}, epsilon: {epsilon}")
            print(
                f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}"
            )  # Show the reward of the last "SHOW_EVERY" episodes

            show = True
        else:
            show = False

        episode_reward = 0
        for i in range(NUM_STEPS_PER_EPISODE):
            observation = (player - food, player - enemy)

            if np.random.random() > epsilon:
                # Exploit
                action = np.argmax(q_table[observation])
            else:
                # Explore
                action = np.random.randint(0, 4)

            player.action(action)

            # Maybe later for complicated movement
            # enemy.move()
            # food.move()

            if player.x == enemy.x and player.y == enemy.y:
                reward = -ENEMY_PENALTY
            elif player.x == food.x and player.y == food.y:
                reward = FOOD_REWARD
            else:
                reward = -MOVE_PENALTY

            new_observation = (player - food, player - enemy)
            max_future_q = np.max(q_table[new_observation])
            current_q = q_table[observation][action]

            if reward == FOOD_REWARD:
                new_q = FOOD_REWARD
            elif reward == -ENEMY_PENALTY:
                new_q = -ENEMY_PENALTY
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                    reward + DISCOUNT * max_future_q
                )

            q_table[observation][action] = new_q

            if show:
                env = np.zeros((AREA_SIZE, AREA_SIZE, 3), dtype=np.uint8)

                # For Array is [y][x] ==> eg) [[1,2,3,4], [5,6,7,8]]
                env[food.y][food.x] = color_dict[FOOD_ID]
                env[player.y][player.x] = color_dict[PLAYER_ID]
                env[enemy.y][enemy.x] = color_dict[ENEMY_ID]

                img = Image.fromarray(env, "RGB")
                img = img.resize((300, 300))

            episode_reward += reward
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                # When the game is over, break the loop
                break

        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

    # Convolve is for smoothing the graph
    moving_avg = np.convolve(
        episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid"
    )

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")

    plt.show()
    plt.savefig("rewards.png")
