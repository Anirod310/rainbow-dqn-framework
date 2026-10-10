import gymnasium as gym
import ale_py
from .preprocessors import AtariPreprocessor
from .models.complete_model import CompleteModel
from .agent.agent import Agent
from.replay_buffer.replay_buffer import ReplayBuffer
from .learning.learner import Learner
import torch


class Factory():
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.environment = None
        self.preprocessor = None

        self.online_model = None
        self.target_model = None

        self.agent = None

        self.mini_batch = None

    def create_environment(self):
        gym.register_envs(ale_py)
        self.environment = gym.make(self.config['environment']['name'], repeat_action_probability=0)
        return self.environment

    def create_preprocessor(self):
        if self.config['preprocessing']['enabled']:
            self.preprocessor = AtariPreprocessor(frame_stack_size=self.config['preprocessing']['frame_stack_size'], 
                                            target_image_size=self.config['preprocessing']['target_image_size'])
            return self.preprocessor
        
    def create_online_model(self):
        input_dim = (
            self.preprocessor.frame_stack_size,
            *self.preprocessor.target_image_size
        ) #To Solve : case when preprocessor is disabled

        output_dim = self.environment.action_space.n

        self.online_model = CompleteModel(input_dim, output_dim)

        return self.online_model

    def create_target_model(self):
        input_dim = (
            self.preprocessor.frame_stack_size,
            *self.preprocessor.target_image_size
        ) #To Solve : case when preprocessor is disabled

        output_dim = self.environment.action_space.n

        self.target_model = CompleteModel(input_dim, output_dim)
        self.target_model.load_state_dict(self.online_model.state_dict())

        return self.target_model

    def create_agent(self):
        self.agent = Agent(self.online_model,
                           self.environment.action_space,
                           self.config['agent']['epsilon_start'],
                           self.device)


        return self.agent

    def create_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.config)

        return self.replay_buffer

    def create_learner(self):
        self.learner = Learner(self.agent, self.target_model, self.config, self.device)

        return self.learner
        
        


