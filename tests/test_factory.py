from rainbow.factory import Factory
from config import config

factory = Factory(config)

env = factory.create_environment()

print(env)
print("Environment spec:", env.spec)
print("Frameskip:", env.unwrapped._frameskip)

obs, info = env.reset()

frames_before = env.unwrapped.ale.getEpisodeFrameNumber()

action = env.action_space.sample()
env.step(action)

frames_after = env.unwrapped.ale.getEpisodeFrameNumber()

print("Frames advanced:", frames_after - frames_before)


