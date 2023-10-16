from .models import Model
from .modifiers import Modifier
from .terminations import Termination

import torch
from torch.distributions.categorical import Categorical

from typing import Callable

class Sampler:
    def __init__(self, model: Model, modifiers: list[Modifier], terminations: list[Termination]):
        self.model = model
        self.modifiers = modifiers
        self.terminations = terminations

    def generate(self, text: str, callback: Callable[[str], None]):
        raise NotImplementedError()

class MultinomialSampler(Sampler):
    def __init__(self, model: Model, modifiers: list[Modifier], terminations: list[Termination]):
        super().__init__(model, modifiers, terminations)

    def generate(self, text: str, callback: Callable[[str], None]):
        tokens = self.model.encode(text)
        while not any(termination.terminate(tokens) for termination in self.terminations):
            dist = self.model(tokens)

            for modifier in self.modifiers:
                dist = modifier(dist)

            multi = Categorical(probs=dist.probs[-1:,:])
            next_token = multi.sample()

            tokens = torch.cat((tokens, next_token), dim=-1)
            text = self.model.decode(tokens)
            callback(text)