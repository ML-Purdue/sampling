import torch
import torch.functoinal as F

class Distribution:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    @property
    def probs(self):
        return F.softmax(self.logits, dim=-1)
