import torch.nn as nn
from ..interface_model import BaseModel

class CNNBackbone(BaseModel):
    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = 512
        self.model = nn.Sequential(
                              nn.Conv2d(input_dim[0], 32, kernel_size=(8,8), stride=4),
                              nn.ReLU(),
                              nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
                              nn.ReLU(),
                              nn.Conv2d(64, 64, kernel_size=(3,3), stride=1),
                              nn.ReLU(),
                              nn.Flatten(),
                              nn.Linear(3136, self.output_dim), #Only works for 84x84 images since we hardcode the shape here
                              nn.ReLU()
                            )

    def forward(self, x):
        x = x/255
        x = self.model(x)
        return x

