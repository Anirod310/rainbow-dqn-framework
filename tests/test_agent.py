from rainbow.factory import Factory
from config import config


factory = Factory(config)

env = factory.create_environment()
preprocessor = factory.create_preprocessor()
model = factory.create_model()
agent = factory.create_agent()

obs, info = env.reset()
processed_obs = preprocessor.reset(obs)

action = agent.select_action(processed_obs)

obs, reward, terminated, truncated, info = env.step(action)

processed_obs = preprocessor.process(obs)


print("Raw observation:", obs.shape)
print("Processed observation:", processed_obs.shape)
print("Action:", action)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

