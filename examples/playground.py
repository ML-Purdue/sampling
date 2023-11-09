import gc
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)


from sampling.models import HFModel
from sampling.modifiers import TopK, TopP, Temperature, ForbiddenTokens
from sampling.terminations import LengthTermination
from sampling.samplers import MultinomialSampler

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using device: {device}")

# model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1').to(device)
# tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

#model_name = "EleutherAI/pythia-160m"
model_name = "Facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

hf_model = HFModel(model, tokenizer, device)

# forbidden_tokens = [hf_model.encode("<|endoftext|>")[0]]
forbidden_tokens = [hf_model.encode("</s>")[0]]
# forbidden_tokens = []

modifiers = [ForbiddenTokens(forbidden_tokens)]
# modifiers = []
terminations = [LengthTermination(750)]
# terminations = []

sampler = MultinomialSampler(hf_model, modifiers, terminations)

start_text = """The quick brown fox jumps over the"""

import psutil
import tracemalloc
tracemalloc.start()

def main():
    outputs = sampler.generate(start_text)
    
    print(start_text, end="")
    numTokensGenerated = 0
    for new_text in outputs:
        print(new_text, end="")
        numTokensGenerated += len(new_text.split(" "))
        sys.stdout.flush()

        if numTokensGenerated % 50 == 0:
            print("")
            print("Torch used memory: ", torch.cuda.memory_allocated() / 1024 / 1024)
            print(f"Torch max memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            mem = psutil.virtual_memory()
            print(f"Avaliable memory: {mem.available / 1024 / 1024} MB")

            
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            print("[ Top 10 ignoring tracemalloc]")
            for stat in top_stats[:10]:
                if "tracemalloc" in str(stat):
                    continue
                print(stat)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**3}KB; Peak was {peak / 10**3}KB; Diff = {(peak - current) / 10**3}KB")
            unreachable = gc.collect()
            print(f"Unreachable objects: {unreachable}")

    print("")
    print(f"Generated {numTokensGenerated} tokens")

def main_hf():
    model_inputs = tokenizer([start_text], return_tensors="pt").to(device)
    new_ids = model.generate(**model_inputs, max_new_tokens=1000, min_length=800, do_sample=True)
    print("".join(tokenizer.batch_decode(new_ids[0])))

main()
print("---------------------------")
# main_hf()
