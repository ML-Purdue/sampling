from .models import Model
import torch

class Termination:
    def __init__(self):
        pass
    
    def terminate(self, tokens: torch.Tensor):
        raise NotImplementedError()
    
class LengthTermination(Termination):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def terminate(self, tokens: torch.Tensor):
        return tokens.size(-1) >= self.length
