from sampling.models import HFModel
from sampling.modifiers import TopK, TopP, Temperature, ForbiddenTokens
from sampling.terminations import LengthTermination
from sampling.samplers import MultinomialSampler

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1').to(device)
# tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m').to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

hf_model = HFModel(model, tokenizer, device)

# forbidden_tokens = [hf_model.encode("<|endoftext|>")[0]]
forbidden_tokens = [hf_model.encode("</s>")[0]]
# forbidden_tokens = []

modifiers = [ForbiddenTokens(forbidden_tokens), TopK(20), Temperature(0.9)]
terminations = [LengthTermination(200)]

sampler = MultinomialSampler(hf_model, modifiers, terminations)

start_text = "The quick brown fox jumps over the"

def main():
    outputs = sampler.generate(start_text)

    print(start_text, end="")
    import sys
    for new_text in outputs:
        print(new_text, end="")
        sys.stdout.flush()

    print("")

def main_hf():
    model_inputs = tokenizer([start_text], return_tensors="pt").to(device)
    new_ids = model.generate(**model_inputs, max_new_tokens=1000, min_length=800, do_sample=True)
    print("".join(tokenizer.batch_decode(new_ids[0])))

main()
print("---------------------------")
# main_hf()
