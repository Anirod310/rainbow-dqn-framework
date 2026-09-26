from rainbow.factory import Factory
from config import config
from rainbow.agent.agent import Agent


factory = Factory(config)

env = factory.create_environment()

preprocessor = factory.create_preprocessor()
