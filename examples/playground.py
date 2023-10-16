# Testing script, not intended for production use


from sampling.models import HFModel
from sampling.modifiers import TopK, TopP, Temperature, ForbiddenTokens
from sampling.terminations import LengthTermination
from sampling.samplers import MultinomialSampler

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

hf_model = HFModel(model, tokenizer)

forbidden_tokens = [hf_model.encode("<|endoftext|>")[0]]

modifiers = [ForbiddenTokens(forbidden_tokens), Temperature(1.0), TopP(0.9)]
terminations = [LengthTermination(200)]

sampler = MultinomialSampler(hf_model, modifiers, terminations)

start_text = "The quick brown fox"

print(start_text, end="")
outputs = sampler.generate(start_text)

import sys
for new_text in outputs:
    print(new_text, end="")
    sys.stdout.flush()

print("")

