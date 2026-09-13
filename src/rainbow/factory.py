import gymnasium as gym
import ale_py
from config import config

gym.register_envs(ale_py)

env = gym.make(config['environment']['name'])