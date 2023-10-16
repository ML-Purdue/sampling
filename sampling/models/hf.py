import torch
import transformers

from .. import Distribution

class HFModel:
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> Distribution:
        logits = self.model(input_ids.unsqueeze(0), **kwargs).logits[0]
        return Distribution(logits=logits)

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors='pt').input_ids[0]

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens)