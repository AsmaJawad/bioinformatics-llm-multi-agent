# pip install git+https://github.com/huggingface/transformers.git # TODO: merge PR to main
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "hemanthkari/llama-3.1-pro-coder-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)


def getCoderAI(usersPrompt):

    prompt = usersPrompt
    messages = [
        {
            "role": "system",
            "content": "You are an expert Python developer. Output ONLY valid Python code from the beginning. No explanations.",
        },
        {"role": "user", "content": f"Write a Python function for: {prompt}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    inputs = inputs.to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    return response
    # just_code = full_text[len(instructionSet):].strip()
    # clean_code = just_code.split('\n\n')[0].strip()
    # return clean_code


def main():
    # Main loop that runs until user breaks with command
    while 1:
        userPrompt = input("Enter a prompt (python function)\n Type `Exit` to quit\n")
        if userPrompt == "Exit":
            break

        # user does not break now ask writer
        ai_globals = {}
        # writerAnswer = getWriterAI(userPrompt)
        coderAnswer = getCoderAI(userPrompt)
        # since the llm ouput includes the `` character, we have to clean that
        lines = coderAnswer.splitlines()
        clean_answer = "\n".join(lines[1:-1])
        print("clean\n", clean_answer)
        
        try:
            print(f"Coder Answers\n {clean_answer}")
            print("SYSTEM OUTPUT")
            exec(clean_answer, ai_globals)
        except Exception as e:
            print(f"The ai code ad an error: {e}")


if __name__ == "__main__":
    main()
