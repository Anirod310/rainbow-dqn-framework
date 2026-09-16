import gymnasium as gym
import ale_py
from ...config import config
from preprocessors import AtariPreprocessor


def create_environment():
    gym.register_envs(ale_py)
    env = gym.make(config['environment']['name'])
    return env

def create_preprocessor():
    if config['preprocessing']['enabled']:
        preprocessor = AtariPreprocessor(frame_stack_size=config['preprocessing']['frame_stack_size'], 
                                         target_image_size=config['preprocessing']['target_image_size'])
        return preprocessor
    
