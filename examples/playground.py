# Testing script, not intended for production use


from sampling.models import HFModel
from sampling.modifiers import TopK, TopP, Temperature, ForbiddenTokens
from sampling.terminations import LengthTermination
from sampling.samplers import MultinomialSampler

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

hf_model = HFModel(model, tokenizer)

# forbidden_tokens = [hf_model.encode("<|endoftext|>")[0]]
forbidden_tokens = [hf_model.encode("</s>")[0]]

modifiers = [ForbiddenTokens(forbidden_tokens), Temperature(1.1)]
terminations = [LengthTermination(200)]

sampler = MultinomialSampler(hf_model, modifiers, terminations)

start_text = "The quick brown fox jumps over the lazy dog."

def main():
    print(start_text, end="")
    outputs = sampler.generate(start_text)

    import sys
    for new_text in outputs:
        print(new_text, end="")
        sys.stdout.flush()

    print("")


main()
# import cProfile
# cProfile.run('main()', sort='cumtime')