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
3. COLUMN / FIELD EXTRACTION (CRITICAL for CSV, TSV, VCF):
   - If the snippet is a CSV or TSV, read the header row and extract the EXACT column names verbatim (e.g., 'control_1', 'treatment_1', 'gene_name').
   - If the snippet is a VCF, note the standard columns present (e.g., 'CHROM', 'POS', 'QUAL', 'INFO', 'FILTER') and any sample columns after FORMAT.
   - Every 'action_desc' in the execution_plan that references a column MUST use the actual column name in single quotes (e.g., "Average df['control_1'], df['control_2'], df['control_3']"). NEVER invent plausible-sounding column names like 'sample', 'control_counts', or 'treatment_counts' — only use names that literally appear in the snippet.
4. CLASSIFICATION: Use the provided Enums to strictly categorize the 'biological_entity' and 'primary_format'.
5. TRANSFORMATION: Select all relevant 'math_transformations' needed to reach the goal.
6. EXECUTION PLAN: Break the task into granular 'execution_step' objects. Each step must be technically specific for the Coder, including the exact column names or field names it must reference. Tools required for each execution step MUST be valid python libraries STRICTLY (e.g. Never recommend R, use pandas/numpy instead).

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

KNOWN API CHANGES (CRITICAL — generated code WILL break if you ignore these):
- Biopython >= 1.80: 'Bio.SeqUtils.GC' has been REMOVED. Use 'from Bio.SeqUtils import gc_fraction' instead, or calculate manually: (seq.count('G') + seq.count('C')) / len(seq).
- Do NOT use 'from Bio.SeqUtils import GC' — it will cause an ImportError.
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

    ---

    Example User Input: For each gene in expression_counts.csv, compare the average of the
    three control samples to the average of the three treatment samples and tell me whether
    it went UP, DOWN, or stayed the SAME and by how many times.

    Example Data from the expression_counts.csv file:
    gene_id,gene_name,length,control_1,control_2,control_3,treatment_1,treatment_2,treatment_3
    ENSG00000141510,TP53,1182,320,335,310,1450,1520,1480
    ENSG00000012048,BRCA1,5592,180,175,190,165,170,180
    ENSG00000076242,MLH1,2524,420,410,435,195,210,200

    Perfect JSON Output:
    {
    "chain_of_thought": "The CSV is a wide-format expression matrix with one row per gene. The header row shows 9 columns: 'gene_id', 'gene_name', 'length', and two groups of three replicates each ('control_1', 'control_2', 'control_3' and 'treatment_1', 'treatment_2', 'treatment_3'). I must reference these exact column names in every step — I will NOT invent a 'sample' column or a 'control_counts' column. For each gene I will compute the mean of the three control columns and the mean of the three treatment columns, then the fold change treatment_mean / control_mean, and classify as UP / DOWN / SAME.",
    "biological_entity": "RNA",
    "primary_format": "LONG_FORMAT",
    "identifier": {
        "detected_type": "Ensembl Gene ID",
        "example_id": "ENSG00000141510",
        "db_source": "Ensembl",
        "requires_translation": false,
        "target_id": null
    },
    "transformations": ["normalization", "custom"],
    "execution_plan": [
        {
        "step_id": 1,
        "task_name": "Load wide-format expression matrix",
        "action_desc": "Read 'input/expression_counts.csv' with pandas.read_csv. The columns are exactly: 'gene_id', 'gene_name', 'length', 'control_1', 'control_2', 'control_3', 'treatment_1', 'treatment_2', 'treatment_3'.",
        "tools_req": ["pandas"]
        },
        {
        "step_id": 2,
        "task_name": "Compute per-group means",
        "action_desc": "Add df['control_mean'] = df[['control_1','control_2','control_3']].mean(axis=1) and df['treatment_mean'] = df[['treatment_1','treatment_2','treatment_3']].mean(axis=1). Use these exact column names — do NOT invent a 'sample' column.",
        "tools_req": ["pandas"]
        },
        {
        "step_id": 3,
        "task_name": "Classify and report fold change",
        "action_desc": "Compute df['fold_change'] = df['treatment_mean'] / df['control_mean']. Label each gene UP if fold_change > 1.5, DOWN if fold_change < 0.67, else SAME. Sort by fold_change descending and print gene_name with its control_mean, treatment_mean, label, and 'x' multiplier.",
        "tools_req": ["pandas"]
        }
      ],
    "correct_output": "For each gene, print the gene_name, control mean, treatment mean, UP/DOWN/SAME label, and how many times it changed, sorted from biggest increase to biggest decrease."
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