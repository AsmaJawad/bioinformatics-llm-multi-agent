# The "Orchestrator" that runs the primary loop. 
# It captures results from the sandbox, 
# feeds them back to the Writer for final clinical 
# summaries, and manages the overall flow.
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from agents import get_writer_config, get_coder_config, format_writer_user_prompt
from sandbox import execute_in_sandbox
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("output.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush() # This ensures it saves even if it crashes!

    def flush(self):
        pass

sys.stdout = Logger()

model_id = "hemanthkari/llama-3.1-pro-coder-v1"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
    low_cpu_mem_usage=True
)

def call_llama(config: dict, user_input: str) -> str:
    
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")    
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=2048, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    torch.cuda.empty_cache() 
    return response

def run_pipeline(user_query, target_file):
    
    try:
        with open(target_file, 'r') as f:
            snippet = f.read(1000)
        print(f"Successfully read snippet from {target_file}")
    except FileNotFoundError:
        print(f"ERROR: '{target_file}' not found in the current folder!")
        return

    writer_config = get_writer_config()
    writer_input = format_writer_user_prompt(user_query, snippet)
    blueprint_json = call_llama(writer_config, writer_input)

    print(f"\n--- Architect Blueprint ---\n{blueprint_json}\n")

    # Implementation
    coder_config = get_coder_config()
    
    coder_input = (
    f"Write a concise UNIVERSAL bash script for this blueprint. "
    f"1. SETUP: Start with 'pip install pandas numpy biopython seaborn matplotlib'. "
    f"2. BASH WRAPPER: Use 'python3 << 'EOF'' to start the Python portion. "
    f"3. HARDCODE FILENAME: Use '{target_file}' directly in the Python code. "
    f"4. SNIFFER & LOGIC: In Python, import io and detect format (VCF/CSV/FASTA). "
    f"   VCF LOADING: Use this EXACT logic: "
    f"   with open('{target_file}', 'r') as f: "
    f"   lines = [l for l in f if not l.startswith('##')] "
    f"   df = pd.read_csv(io.StringIO(''.join(lines)), sep='\\t') "
    f"5. HEADER CLEANING: Run 'df.columns = [c.lstrip(\"#\") for c in df.columns]'. "
    f"6. VALIDATION: Print 'df.columns' to the log. "
    f"7. ROBUST DP PARSING: Extract DP from INFO using regex to get the numeric part only. "
    f"   Example: df['DP'] = df['INFO'].str.extract(r'DP=(\\\\d+)').astype(float). "
    f"8. FILTERING: Apply (QUAL > 30) and (DP > 10). Keep only SNPs (len(REF)==1). "
    f"9. TERMINATION: End the Python portion with a single line: 'EOF'. "
    f"10. BASH CLEANUP: After 'EOF', add shell cleanup like 'rm -rf *.tmp'. "
    f"11. CONSTRAINTS: Do NOT put shell commands inside Python. Output ONLY code:\n{blueprint_json}"
    )
    
    bash_script = call_llama(coder_config, coder_input)

    bash_script = bash_script.replace("```bash", "").replace("```", "").strip()
    result = execute_in_sandbox(bash_script, target_file)
    print(result)


if __name__ == "__main__":
    USER_QUERY = (    
    "Calculate the GC content for each sequence in the sequences.fasta "
    "FASTA file and identify  sequences with GC content above 60%. "
    "Also find all EcoRI restriction sites in these high-GC sequences."
    )
    run_pipeline(USER_QUERY, "sequences.fasta")