import gymnasium as gym
import ale_py
from .preprocessors import AtariPreprocessor


class Factory():
    def __init__(self, config):
        self.config = config

    def create_environment(self):
        gym.register_envs(ale_py)
        env = gym.make(self.config['environment']['name'])
        return env

    def create_preprocessor(self):
        if self.config['preprocessing']['enabled']:
            preprocessor = AtariPreprocessor(frame_stack_size=self.config['preprocessing']['frame_stack_size'], 
                                            target_image_size=self.config['preprocessing']['target_image_size'])
            return preprocessor
        
