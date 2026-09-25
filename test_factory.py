from rainbow.factory import Factory
from config import config

factory = Factory(config)

env = factory.create_environment()

preprocessor = factory.create_preprocessor()

obs, info = env.reset()

processed_obs = preprocessor.reset(obs)

action = env.action_space.sample()

obs, reward, terminated, truncated, info = env.step(action)

processed_obs = preprocessor.process(obs)

print(processed_obs.shape)
print(processed_obs.dtype)
print(processed_obs.min())
print(processed_obs.max())