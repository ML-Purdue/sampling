# Testing script, not intended for production use

from sampling.models import HFModel
from sampling.modifiers import TopK, TopP, Temperature
from sampling.terminations import LengthTermination
from sampling.samplers import MultinomialSampler

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

hf_model = HFModel(model, tokenizer)
modifiers = [Temperature(0.9)]
terminations = [LengthTermination(200)]

sampler = MultinomialSampler(hf_model, modifiers, terminations)

def callback(text):
    print(text)

sampler.generate("Hello! How", callback)

