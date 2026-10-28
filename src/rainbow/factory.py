import gymnasium as gym
import ale_py

from .models.complete_model import CompleteModel
from.models.backbone.mlp_backbone import MLPBackbone
from.models.backbone.cnn_backbone import CNNBackbone

from .agent.agent import Agent

from .replay_buffer.replay_buffer import ReplayBuffer

from .learning.learner import Learner

from.environment.wrappers.fire_reset import FireResetWrapper

import torch
import random
import numpy as np


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


    def set_random_seed(self):
        seed = self.config['environment']['random_seed']

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        return seed 


    def create_environment(self):
        gym.register_envs(ale_py)
        self.environment = gym.make(self.config['environment']['name'], frameskip=self.config['environment']['frameskip_env'], repeat_action_probability = 0, render_mode=self.config['environment']['render_mode'])
        if self.config['preprocessing']['enabled'] and (self.config['preprocessing']['type'] == 'atari'):
            self.environment = gym.wrappers.AtariPreprocessing(self.environment,
                                                        frame_skip=self.config['environment']['frameskip_wrap'],
                                                        screen_size=tuple(self.config['preprocessing']['image_size']),
                                                        grayscale_obs=True,
                                                        scale_obs=False)
            self.environment = gym.wrappers.FrameStackObservation(self.environment, stack_size=self.config['preprocessing']['frame_stack_size'])
        return self.environment

    def create_manual_play_environment(self):
        gym.register_envs(ale_py)
        self.environment = gym.make(self.config['environment']['name'], obs_type="rgb", repeat_action_probability=0.0, render_mode="rgb_array")
        return self.environment

    def create_fire_reset_wrapper(self):
        fire_reset_wrapper = FireResetWrapper(self.environment)
        return fire_reset_wrapper

    def create_preprocessor(self):
        if self.config['preprocessing']['enabled']:
            ...#Not used anymore since we use gym wrappers
            '''self.preprocessor = AtariPreprocessor(frame_stack_size=self.config['preprocessing']['frame_stack_size'], 
                                            target_image_size=self.config['preprocessing']['target_image_size'])
            '''
            return self.preprocessor
        
    def create_online_model(self):
        if self.config['preprocessing']['enabled'] : 
            input_dim = (
                self.config['preprocessing']['frame_stack_size'],
                *self.config['preprocessing']['image_size'])
        else : 
            shape = self.environment.observation_space.shape
            input_dim = int(shape[0])

        output_dim = self.environment.action_space.n

        if self.config['environment']['type'] == "atari":
            backbone = CNNBackbone
        else : 
            backbone = MLPBackbone

        self.online_model = CompleteModel(input_dim, output_dim, backbone)

        return self.online_model

    def create_target_model(self):
        if self.config['preprocessing']['enabled'] : 
            input_dim = (
                self.config['preprocessing']['frame_stack_size'],
                *self.config['preprocessing']['image_size'])

        else : 
            shape = self.environment.observation_space.shape
            input_dim = int(shape[0])

        output_dim = self.environment.action_space.n

        if self.config['environment']['type'] == "atari":
            backbone = CNNBackbone
        else : 
            backbone = MLPBackbone

        self.target_model = CompleteModel(input_dim, output_dim, backbone)

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
        
        


