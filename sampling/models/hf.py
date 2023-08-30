import transformers

from .. import Distribution

class HFModel:
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor) -> Distribution:
        logits = self.model(input_ids, **kwargs).logits
        return Distribution(logits=logits)

    def decode(self, input_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(input_ids)