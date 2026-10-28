import torch.nn as nn
from ..interface_model import BaseModel

class MLPBackbone(BaseModel):
    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = 128
        self.model = nn.Sequential(
                            nn.Linear(input_dim, 128),
                            nn.ReLU(),
                            nn.Linear(128, 128),
                            nn.ReLU(),
                            )

    def forward(self, x):
        x = self.model(x)
        return x
