import torch.nn as nn
from ..interface_model import BaseModel
import torch


class DuelingHead(BaseModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.value_stream = nn.Linear(input_dim, 1)
        self.advantage_stream = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        x = value + (advantage - torch.mean(advantage,
                                            dim=1,
                                            keepdim=True))
        return x


