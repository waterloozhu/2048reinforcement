from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import numpy as np
from enviromentfor2048 import Game2048Env  # Ensure correct import path
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.pyplot as plt

# Function to create the environment
def create_environment():
    return Game2048Env()

def evaluate_model(model, num_episodes=10):
    env = create_environment()
    max_tiles = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if done:
                max_tile = np.max(obs)
                max_tiles.append(max_tile)
    env.close()
    return max_tiles

model_path = "dqn_2048.zip"
env = None
if os.path.exists(model_path):
    temp_env = create_environment()
    model = DQN.load(model_path, env=temp_env)
    print("Loaded existing model")
    env = temp_env
else:
    env = make_vec_env(create_environment, n_envs=1)
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.1, buffer_size=10000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, optimize_memory_usage=True, target_update_interval=500, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10)
    print("Created new model")

total_timesteps = 5000


model.learn(total_timesteps=total_timesteps)
model.save(model_path)

env.close()
# Evaluate the model
'''
max_tiles = evaluate_model(model, num_episodes=100)

# Plotting
plt.plot(max_tiles)
plt.ylabel('Maximum Tile Value')
plt.xlabel('Episode')
plt.title('Model Improvement Over Time')
plt.show()

# Save the model for future use
model.save(model_path)
print(f"Model saved to {model_path}")
env.close()'''