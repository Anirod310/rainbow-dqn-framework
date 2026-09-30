import gymnasium as gym
import ale_py
from .preprocessors import AtariPreprocessor
from .models.complete_model import CompleteModel


class Factory():
    def __init__(self, config):
        self.config = config
        self.environment = None
        self.preprocessor = None
        self.model = None

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
        )

        output_dim = self.environment.action_space.n

        self.model = CompleteModel(input_dim, output_dim)

        return self.model


