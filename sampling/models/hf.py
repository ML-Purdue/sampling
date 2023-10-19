import torch
import transformers

from .. import Distribution

class HFModel:
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else "cpu"

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> Distribution:
        logits = self.model(input_ids.unsqueeze(0).to(self.device), **kwargs).logits.cpu().detach()[0,-1,:]

        return Distribution(logits=logits)

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors='pt').input_ids[0]

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens)
