# Contains the System Prompts for the Writer and Coder LLMs. 
# It instructs the Writer to use the JSON schema 
# and the Coder to generate sandboxed, error-handled code.


#TODO: try running this first and analyze speed. reduce the number of lines read by writer from the file to 3 lines
# writer needs to know its truncated file not full
# consider reducing instructions as LLMs are designed to do many of the functionalities 
# in-context learning 

import os
from typing import List, Optional, Dict, Any

# Importing the classes from your schemas.py to maintain the structural contract
from schemas import (
    sys_blueprint, 
    identifier_type, 
    execution_step, 
    math_transformations, 
    bio_entity, 
    file_format
)

# =================================================================
# WRITER SYSTEM PROMPT (The Architect)
# =================================================================

identifier_fields = ", ".join(identifier_type.model_fields.keys()) # Help the model find the identifier_type schema and strictly adhere to that structure

exec_plan_key = list(sys_blueprint.model_fields.keys())[5] # The plan in sys_blueprint for coder instructions
tools_key = list(execution_step.model_fields.keys())[3] # Dynamically gets 'tools_req' from execution plan
output_desc = sys_blueprint.model_fields['correct_output'].description # Gets the output field created in the JSON blueprint

WRITER_SYSTEM_PROMPT = f"""
You are an expert Bioinformatics Architect. Your goal is to interpret a researcher's request and provide a technical execution blueprint in JSON format.

REASONING PROTOCOL:
1. THINKING: Use the 'chain_of_thought' field to analyze the biological context in English. 
2. DATA INSPECTION: Look at the provided 'DATA SNIPPET' to determine the 'identifier_type'.
   - Identify the source database (e.g., Ensembl, NCBI, UniProt).
   - Detect the specific ID pattern (e.g., ENSG... for Ensembl Gene, rs... for dbSNP).
   - Determine if 'requires_translation' is True (e.g., if the user wants to use a tool that requires Entrez IDs but provided Gene Symbols).
3. CLASSIFICATION: Use the provided Enums to strictly categorize the 'biological_entity' and 'primary_format'.
4. TRANSFORMATION: Select all relevant 'math_transformations' needed to reach the goal.
5. EXECUTION PLAN: Break the task into granular 'execution_step' objects. Each step must be technically specific for the Coder.

STRICT SCHEMA DEFINITION:
You must output JSON that matches {sys_blueprint.__name__} structure:
- biological_entity: Must be one of { [e.value for e in bio_entity] }
- primary_format: Must be one of { [f.value for f in file_format] }
- transformations: A list containing values from { [m.value for m in math_transformations] }
- identifier: A dictionary with fields: {identifier_fields}
    - When filling out the identifier block, look at the data head. If you see ENSG..., the db_source is Ensembl. If you see rs..., it is dbSNP.

OUTPUT RULES:
- Return ONLY valid JSON.
- Ensure the JSON strictly adheres to the {sys_blueprint.__name__} schema.
- Do not include any text, markdown (like ```json), or explanations outside the JSON block.
"""

# =================================================================
# CODER SYSTEM PROMPT (The Programmer)
# =================================================================
CODER_SYSTEM_PROMPT = f"""
You are a Senior Systems Programmer specialized in Bioinformatics and Python.
Your goal is to implement the provided {sys_blueprint.__name__} into a self-contained execution script.

STRICT CONSTRAINTS:
1. OUTPUT FORMAT: Your response must be a single code block containing a BASH SCRIPT.
2. SANDBOX COMPLIANCE: The Bash script must handle its own environment. It MUST:
   - Use 'set -e' to exit immediately on errors.
   - Create a local virtual environment: 'python3 -m venv venv'
   - Activate the environment: 'source venv/bin/activate'
   - Install all libraries listed in the {tools_key} field of every execution step in the {exec_plan_key} array using pip.
   - WRITING THE CODE: Use a 'cat << "EOF" > solution.py' block to write the Python logic.
   - EXECUTION: Run the generated Python script: 'python3 solution.py'.
   - OUTPUT: Ensure the script prints the exact {output_desc} to stdout.
   - CLEANUP: Deactivate and remove the 'venv' folder before exiting.
3. ERROR HANDLING: Use try-except blocks in the Python code and print clear error messages to stderr to help the Orchestrator.
"""

# =================================================================
# HELPER FUNCTIONS
# =================================================================

def get_few_shot_examples() -> str:
    """
    Teaches the Writer how to request specific Biopython logic 
    (like EcoRI searches) in the execution plan.

    """
    return """
    Example User Input: Calculate the GC content for each sequence in the sequences.fasta FASTA file 
    and identify  sequences with GC content above 60%. 
    Also find all EcoRI restriction sites in these high-GC sequences.

    Example Data from the sequences.fasta file:
    >seq1 
    ATCGATCGATCGAATTCGCGCGCGCGATATATATGAATTCGCGCGC 
    >seq2 
    ATATATATATATATGAATTCATATATATATAT 
    >seq3 
    GCGCGCGCGCGCGAATTCGCGCGCGCGCGCGC 
    >seq4 
    TTTTAAAATTTTAAAAGAATTCTTTTAAAA

    Perfect JSON Output:
    {
    "chain_of_thought": "The user wants to analyze sequence headers and content in a FASTA file. I will calculate GC content, filter by 60%, and then search for the EcoRI motif (GAATTC). IDs are local headers; no translation needed.",
    "biological_entity": "DNA",
    "primary_format": "FASTA",
    "identifier": {
        "detected_type": "FASTA Header",
        "example_id": "seq1",
        "db_source": "Local File",
        "requires_translation": false,
        "target_id": null
    },
    "transformations": ["filtering", "custom"],
    "execution_plan": [
        {
        "step_id": 1,
        "task_name": "Sequence Analysis & Filtering",
        "action_desc": "Iterate through 'sequences.fasta' using Biopython's SeqIO. Calculate GC content for each record and filter for GC > 60%.",
        "tools_req": ["biopython"]
        },
        {
        "step_id": 2,
        "task_name": "Restriction Mapping",
        "action_desc": "For sequences passing the filter, use Bio.Restriction.EcoRI to find 1-based indices of all GAATTC sites.",
        "tools_req": ["biopython"]
        }
      ],
    "correct_output": "The program should print the ID and GC content of all sequences. For sequences with GC > 60%, it must list the positions of any EcoRI sites found."
    }
    """

def format_writer_user_prompt(user_request: str, data_head: str) -> str:
    """
    Combines the user request and file snippet for the Writer.
    This solves the 'ID Translation' problem by providing visual context.
    """
    return f"""
    USER REQUEST: {user_request}
    
    DATA SNIPPET (First 10 lines of file):
    ---
    {data_head}
    ---
    
    Generate the sys_blueprint JSON based on the request and the identifiers found in the snippet.
    """

def get_writer_config() -> Dict[str, Any]:
    """Returns the settings for the Writer Agent."""
    return {
        "system_prompt": WRITER_SYSTEM_PROMPT + "\n" + get_few_shot_examples(),
        "temperature": 0.1,
        "model": "hemanthkari/llama-3.1-pro-coder-v1"
    }

def get_coder_config() -> Dict[str, Any]:
    """Returns the settings for the Coder Agent."""
    return {
        "system_prompt": CODER_SYSTEM_PROMPT,
        "temperature": 0.0,
        "model": "hemanthkari/llama-3.1-pro-coder-v1"
    }