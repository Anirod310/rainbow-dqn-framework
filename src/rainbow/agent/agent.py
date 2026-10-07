import torch
import numpy as np

class Agent():
    def __init__(self, model, action_space, epsilon, device):
        self.model = model
        self.action_space = action_space
        self.epsilon = epsilon
        self.device = device

    def select_action(self, observation):
        
        if(np.random.rand() < self.epsilon):
            action = np.random.randint(self.action_space.n)
        else:
            observation_tensor = torch.tensor(observation).unsqueeze(0).to(self.device)
            q_values = self.model(observation_tensor)
            action = int(q_values.argmax())
        return action
        

