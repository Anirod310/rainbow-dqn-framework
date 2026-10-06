import gymnasium as gym
import ale_py
from .preprocessors import AtariPreprocessor
from .models.complete_model import CompleteModel
from .agent.agent import Agent
from.replay_buffer.replay_buffer import ReplayBuffer
from .learning.learner import Learner


class Factory():
    def __init__(self, config):
        self.config = config
        self.environment = None
        self.preprocessor = None
        self.model = None
        self.agent = None
        self.mini_batch = None

    def create_environment(self):
        gym.register_envs(ale_py)
        self.environment = gym.make(self.config['environment']['name'])
        return self.environment

    def create_preprocessor(self):
        if self.config['preprocessing']['enabled']:
            self.preprocessor = AtariPreprocessor(frame_stack_size=self.config['preprocessing']['frame_stack_size'], 
                                            target_image_size=self.config['preprocessing']['target_image_size'])
            return self.preprocessor
        
    def create_model(self):
        input_dim = (
            self.preprocessor.frame_stack_size,
            *self.preprocessor.target_image_size
        ) #To Solve : case when preprocessor is disabled

        output_dim = self.environment.action_space.n

        self.model = CompleteModel(input_dim, output_dim)

        return self.model

    def create_agent(self):
        self.agent = Agent(self.model,
                           self.environment.action_space,
                           self.config['agent']['epsilon'])

        return self.agent

    def create_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.config)

        return self.replay_buffer

    def create_learner(self):
        self.learner = Learner(self.agent, self.config)

        return self.learner
        
        


