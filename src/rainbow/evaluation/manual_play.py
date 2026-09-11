import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5', obs_type="rgb", repeat_action_probability=0.0, render_mode="human")

obs, info = env.reset()

done = False

while not done :
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated :
        done = True

env.close()
