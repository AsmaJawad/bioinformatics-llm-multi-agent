# The "Orchestrator" that runs the primary loop. 
# It captures results from the sandbox, 
# feeds them back to the Writer for final clinical 
# summaries, and manages the overall flow.

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from agents import get_writer_config, get_coder_config, format_writer_user_prompt
from sandbox import execute_in_sandbox

if "LLM_API_KEY" in os.environ:
    del os.environ["LLM_API_KEY"]

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("BASE_URL", "https://api.together.xyz/v1"),
    api_key=os.getenv("LLM_API_KEY")
)

def call_llama(config: dict, user_input: str) -> str:
    """Standardizes the call to llama model."""
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": user_input}
        ],
        temperature=config["temperature"],
        max_tokens=2048
    )
    return response.choices[0].message.content
    
def run_pipeline(user_query, target_file):
    
    try:
        with open(target_file, 'r') as f:
            snippet = f.read(1000)
        print(f"✅ Successfully read snippet from {target_file}")
    except FileNotFoundError:
        print(f"❌ ERROR: '{target_file}' not found in the current folder!")
        return
    
    with open(target_file, 'r') as f:
        snippet = f.read(1000)

    # Architect -> Implement -> Execute
    writer_config = get_writer_config()
    writer_input = format_writer_user_prompt(user_query, snippet)
    blueprint_json = call_llama(writer_config, writer_input)

    # Implementation

    coder_config = get_coder_config()
    bash_script = call_llama(coder_config, blueprint_json)

    result = execute_in_sandbox(bash_script)
    
    print("\n--- FINAL BIOINFORMATICS OUTPUT ---")
    print(result)


if __name__ == "__main__":
    USER_QUERY = (
        "Calculate the GC content for each sequence in the sequences.fasta FASTA file "
        "and identify sequences with GC content above 60%. "
        "Also find all EcoRI restriction sites in these high-GC sequences."
    )
    run_pipeline(USER_QUERY, "sequences.fasta")