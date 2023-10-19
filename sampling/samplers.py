from .models import Model
from .modifiers import Modifier
from .terminations import Termination
import gc

import torch

from typing import Callable

class Sampler:
    def __init__(self, model: Model, modifiers: list[Modifier], terminations: list[Termination]):
        self.model = model
        self.modifiers = modifiers
        self.terminations = terminations

    def generate(self, text: str):
        raise NotImplementedError()

class MultinomialSampler(Sampler):
    def __init__(self, model: Model, modifiers: list[Modifier], terminations: list[Termination]):
        super().__init__(model, modifiers, terminations)

    def generate(self, text: str):
        tokens = self.model.encode(text)

        while not any(termination.terminate(tokens) for termination in self.terminations):
            dist = self.model(tokens)

            for modifier in self.modifiers:
                dist = modifier(dist)

            next_token = torch.multinomial(dist.probs, num_samples=1)

            original_length = len(self.model.decode(tokens))

            tokens = torch.cat((tokens, next_token), dim=-1)

            new_text = self.model.decode(tokens)[original_length:]

            yield new_text

            gc.collect()
