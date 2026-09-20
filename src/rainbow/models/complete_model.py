from .cnn_backbone import CNNBackbone
from.q_value_head import QValueHead
import torch.nn as nn


class CompleteModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.backbone = CNNBackbone(input_dim)
        self.head = QValueHead(self.backbone.output_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x

