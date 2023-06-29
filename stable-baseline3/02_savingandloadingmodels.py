import gym
from stable_baselines3 import A2C
import os


models_dir = "./models/A2C"
logdir = "./logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2")
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cuda:2")
"""
# Load model
model_path = f"{models_dir}/250000.zip"
model = PPO.load(model_path, env=env)
"""

TIMESTEPS = 1000
iters = 0
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
