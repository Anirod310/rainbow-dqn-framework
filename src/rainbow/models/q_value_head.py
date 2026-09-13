import torch.nn as nn
from .base_model import BaseModel
import torch


class QValueHead(BaseModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

