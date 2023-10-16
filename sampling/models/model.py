import torch
from .. import Distribution

class Model:
    def __init__(self):
        pass

    def __call__(self, input_ids: torch.Tensor) -> Distribution:
        raise NotImplementedError()
    
    def encode(self, text: str) -> torch.Tensor:
        raise NotImplementedError()

    def decode(self, tokens: torch.Tensor) -> str:
        raise NotImplementedError()