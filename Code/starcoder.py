# pip install git+https://github.com/huggingface/transformers.git # TODO: merge PR to main
from transformers import AutoModelForCausalLM, AutoTokenizer

def getWriterAI(usersPrompt):
	
		writerAI= "bigcode/starcoder2-7b"
		device = "cuda" # for GPU usage or "cpu" for CPU usage

		tokenizer = AutoTokenizer.from_pretrained(writerAI)
		# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(writerAI, device_map="auto")`
		model = AutoModelForCausalLM.from_pretrained(writerAI).to(device)

		# temp = float(input("Enter temp 0.0 - 1"))
		temp = .7
		# get the prompt from the user
		prompt = usersPrompt
		instructionSet = f"\n# Break down instructions and write about the each part of the step by step instructions of how to implement the given function\n# {prompt}\n# End the function after implementation with marker # END_OF_FUNCTION\n\n"
		print(f"Writers Prompt\n{instructionSet}")
		tokenizer.pad_token = tokenizer.eos_token
		inputs = tokenizer(instructionSet, return_tensors="pt", padding=True)
		inputs = {k: v.to(device) for k, v in inputs.items()}
		# temperature -> Controls randomness / creativity (higher values are more creative)
		# top_p -> Nucleus sampling (This is another way to limit randomness. Instead of letting the model choose from all possible next tokens, it only samples from the smallest group of tokens whose combined probability is 90% (0.9).)
		# eos_token_id=tokenizer.eos_token_id -> End-of-sequence marker. This tells the model: “You can stop generating when you reach this special token.”

		outputs = model.generate(**inputs,max_new_tokens=100,do_sample=True, temperature=temp, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
		writerOutput= tokenizer.decode(outputs[0], skip_special_tokens=True)

		marker = "# END_OF_FUNCTION"
		p = writerOutput.split(marker)

		if len(p) >=2:
			finalWriterOutput= marker.join(p[:2])
		else:
			finalWriterOutput = writerOutput
		# if marker in code:
		#     code = code.split(marker)[0] + marker
					
		return finalWriterOutput.strip()

def getCoderAI(usersPrompt):
	
		coderAI= "bigcode/starcoder2-7b"
		device = "cuda" # for GPU usage or "cpu" for CPU usage

		tokenizer = AutoTokenizer.from_pretrained(coderAI)
		# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(coderAI, device_map="auto")`
		model = AutoModelForCausalLM.from_pretrained(coderAI).to(device)

		# temp = float(input("Enter temp 0.0 - 1"))
		temp = .3
		# get the prompt from the user
		prompt = usersPrompt
		instructionSet = (
			f"Task: Write a python function for: {prompt}\n"
			"Constraints: Include necessary imports. "
			"Output ONLY the python code. No Explanations. No text before or after.\n"
			"Code:\n"
		)

		tokenizer.pad_token = tokenizer.eos_token
		inputs = tokenizer(instructionSet, return_tensors="pt", padding=True).to(device)
		# temperature -> Controls randomness / creativity (higher values are more creative)
		# top_p -> Nucleus sampling (This is another way to limit randomness. Instead of letting the model choose from all possible next tokens, it only samples from the smallest group of tokens whose combined probability is 90% (0.9).)
		# eos_token_id=tokenizer.eos_token_id -> End-of-sequence marker. This tells the model: “You can stop generating when you reach this special token.”

		outputs = model.generate(
			**inputs,
			max_new_tokens=350,
			do_sample=True,
			temperature=temp,
			top_p=0.9,
			eos_token_id=tokenizer.eos_token_id
		)
		full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

		just_code = full_text[len(instructionSet):].strip()
		clean_code = just_code.split('\n\n')[0].strip()
		return clean_code
def main():
	# Main loop that runs until user breaks with command
	while 1:
		userPrompt = input("Enter a prompt (python function)\n Type `Exit` to quit\n")
		if userPrompt == "Exit":
			break
			
		# user does not break now ask writer
		ai_globals = {}
		#writerAnswer = getWriterAI(userPrompt)
		coderAnswer = getCoderAI(userPrompt)
		try:
			print(f"Coder Answers\n {coderAnswer}")
			print("SYSTEM OUTPUT")
			exec(coderAnswer, ai_globals)
		except Exception as e:
			print(f"The ai code ad an error: {e}")
if __name__ == "__main__":
	main()
