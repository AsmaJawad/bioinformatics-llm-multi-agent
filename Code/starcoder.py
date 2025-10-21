# pip install git+https://github.com/huggingface/transformers.git # TODO: merge PR to main
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder2-7b"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# get the prompt from the user
prompt = input("Enter prompt (e.g. Write a Python function that adds two numbers)\n")
f_prompt = f"\n# Write a Python function only. Do not include any other languages\n# {prompt}\n# End the function after implementation with marker # END_OF_FUNCTION\n\n"

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(f_prompt, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
# temperature -> Controls randomness / creativity (higher values are more creative)
# top_p -> Nucleus sampling (This is another way to limit randomness. Instead of letting the model choose from all possible next tokens, it only samples from the smallest group of tokens whose combined probability is 90% (0.9).)
# eos_token_id=tokenizer.eos_token_id -> End-of-sequence marker. This tells the model: “You can stop generating when you reach this special token.”

outputs = model.generate(**inputs,max_new_tokens=100,do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)

marker = "# END_OF_FUNCTION"
p = code.split(marker)

if len(p) >=2:
    code = marker.join(p[:2])
else:
    code = code
# if marker in code:
#     code = code.split(marker)[0] + marker

print(code.strip()+ "\n" +marker)