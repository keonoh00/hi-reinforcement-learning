import gym
from stable_baselines3 import A2C

env = gym.make("LunarLander-v2")  # continuous: LunarLanderContinuous-v2
env.reset()

model = A2C("MlpPolicy", env, verbose=1, device="cuda:2")
model.learn(total_timesteps=100000)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        print(rewards)
