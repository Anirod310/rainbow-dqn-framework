import gymnasium as gym
import ale_py
from gymnasium.utils.play import play

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5', obs_type="rgb", repeat_action_probability=0.0, render_mode="rgb_array")

play(env, keys_to_action={('e',): 0, (' ',): 1, ('d',): 2, ('q',): 3})




