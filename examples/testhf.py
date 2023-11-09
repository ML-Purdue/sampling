from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
model_ref = "EleutherAI/gpt-neo-125M"
model = AutoModelForCausalLM.from_pretrained(

    model_ref, device_map="auto", load_in_4bit=True

)

tokenizer = AutoTokenizer.from_pretrained(model_ref, padding_side="left")

model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]