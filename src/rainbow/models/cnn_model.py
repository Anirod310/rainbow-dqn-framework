import torch
import torch.nn.functional as nn
from base_model import BaseModel

class CNNAtari(BaseModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()
    