import torch

class Agent():
    def __init__(self, model, action_space, epsilon):
        self.model = model
        self.action_space = action_space
        self.epsilon = epsilon

    def select_action(self, observation):
        observation_tensor = torch.tensor(observation).unsqueeze(0)
        q_values = self.model(observation_tensor)
        return q_values
        

