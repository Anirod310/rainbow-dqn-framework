from src.rainbow.factory import factory

env = factory.create_environment()

preprocessor = factory.create_preprocessor()

obs, info = env.reset()

processed_obs = preprocessor.reset(obs)

action = env.action_space.sample()

obs, reward, terminated, truncated, info = env.step()

processed_obs = preprocessor.process(obs)

print(f"observation shape : {processed_obs.shape}")