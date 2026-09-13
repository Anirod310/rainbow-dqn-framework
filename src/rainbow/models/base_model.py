from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(ABC):

    @abstractmethod
    def forward(self, x):
        ...
