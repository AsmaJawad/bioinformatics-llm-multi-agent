# Multi-Agent LLM System for Bioinformatics

## Quick Summary

### What It Can Do
- Take a bioinformatics data file (FASTA, VCF, or CSV) and a plain-English request, then **automatically generate, execute, and return results** from a Python analysis script
- Parse FASTA sequences (GC content, restriction sites, sequence stats)
- Filter and summarize VCF variant files (SNPs, quality/depth filtering)
- Normalize and analyze gene expression matrices (TPM, fold change, heatmaps)
- Auto-install any required Python libraries (pandas, biopython, seaborn, etc.) inside an isolated sandbox
- Self-correct by feeding errors back through the pipeline
- Save the generated `solution.py` and any output files (plots, filtered data) to your working directory

### What It Cannot Do
- Generate code in languages other than Python
- Run analyses requiring interactive user input mid-execution
- Install packages that need system-level C compilation (e.g., pysam may fail; pure-Python packages work best)
- Access external APIs or databases at runtime (all analysis is local and offline)
- Handle files larger than what fits in memory — designed for small-to-medium research datasets
- Guarantee correctness — the LLM generates code on the fly, so **always review `solution.py` before trusting results**

### Requirements
- NVIDIA GPU with CUDA support (model runs in 4-bit quantization)
- Python 3.10+
- ~8 GB VRAM minimum

### Usage
```bash
# Interactive mode — guided prompts help you build your query
python3 main.py sequences.fasta

# Direct mode — provide the query inline
python3 main.py variants.vcf "Filter SNPs with quality > 30 and depth > 10"

# CSV expression data
python3 main.py expression_counts.csv
```

### Expected Output
- Results printed to terminal
- `solution.py` — the generated Python script, saved to your directory
- `output.log` — full session log
- Any generated files (plots, filtered datasets) copied back from the sandbox

---

## Detailed Documentation

### Overview

This is a **multi-agent AI system** that translates natural-language bioinformatics requests into executable Python code. It uses a locally-hosted Llama 3.1 model (no cloud APIs) with two specialized agent roles that collaborate through a structured JSON contract.

The system follows a three-stage pipeline:

```
User Request + Data File
        |
        v
  [Writer Agent]  -->  JSON Blueprint (execution plan)
        |
        v
  [Coder Agent]   -->  Bash script containing solution.py
        |
        v
  [Sandbox]        -->  Isolated execution, results returned
```

### Architecture

#### Files

| File | Role |
|------|------|
| `main.py` | **Orchestrator** — CLI entry point, loads the model, runs the Writer/Coder pipeline, handles interactive prompts |
| `agents.py` | **Agent Prompts** — System prompts and few-shot examples for the Writer (Architect) and Coder (Programmer) agents |
| `schemas.py` | **Data Contracts** — Pydantic models that define the JSON blueprint structure (bio entities, file formats, execution steps) |
| `sandbox.py` | **Sandbox Executor** — Creates an isolated temp directory, runs the generated bash script, copies results back, then destroys the sandbox |
| `tools.py` | **External API Tools** — (Placeholder) Wrapper functions for NCBI/MyGene.info to enrich results with biological annotations |

#### Agent Roles

**Writer Agent (The Architect)**
- Receives the user's request and a data snippet (first 1000 chars of the file)
- Produces a structured JSON blueprint (`sys_blueprint`) containing:
  - `chain_of_thought` — reasoning about the biological context
  - `biological_entity` — DNA, RNA, Protein, etc.
  - `primary_format` — FASTA, VCF, CSV, etc.
  - `identifier` — detected ID type, database source, whether translation is needed
  - `transformations` — normalization, filtering, clustering, etc.
  - `execution_plan` — step-by-step instructions with required Python libraries
  - `correct_output` — what the final program should print

**Coder Agent (The Programmer)**
- Receives the JSON blueprint
- Generates a self-contained bash script that:
  - Creates a virtual environment
  - Installs required libraries
  - Writes a `solution.py` file
  - Executes it and prints results
  - Cleans up the virtual environment

#### Sandbox

All generated code runs in a **temporary directory** that is destroyed after execution:
1. A unique temp folder is created
2. The user's data file is copied in
3. The generated bash script runs inside it
4. `solution.py` and any output files are copied back to the working directory
5. The entire temp folder (including any venv) is deleted

This prevents generated code from polluting your environment or leaving artifacts behind.

### Supported File Formats

| Format | Extensions | Typical Analysis |
|--------|-----------|------------------|
| FASTA | `.fasta`, `.fa`, `.fna` | GC content, restriction sites, sequence statistics |
| VCF | `.vcf` | Variant filtering, SNP distribution, quality analysis |
| CSV | `.csv` | Gene expression normalization (TPM), differential expression, heatmaps |
| TSV | `.tsv` | Same as CSV with tab delimiter |

### Interactive Mode

When you run without a query, the system asks three guided questions:

1. **Analysis type** — Preset options based on your file format, or describe your own
2. **Thresholds/filters** — Use sensible defaults or specify custom values
3. **Output preference** — Print to terminal, save to file, or both

### Model

The system uses [`hemanthkari/llama-3.1-pro-coder-v1`](https://huggingface.co/hemanthkari/llama-3.1-pro-coder-v1) loaded with 4-bit NF4 quantization via `bitsandbytes` to fit on consumer GPUs. Both agents share the same model but use different system prompts and temperature settings:
- Writer: `temperature=0.1` (low creativity, structured output)
- Coder: `temperature=0.0` (deterministic code generation)

### Installation

```bash
cd Code/multi_agent_llm_system

# Create and activate virtual environment
python3 -m venv ../venv
source ../venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt
```

Required packages (see `requirements.txt`):
- `torch` (with CUDA)
- `transformers`
- `bitsandbytes`
- `pydantic`
- `accelerate`

### Example Workflows

**1. FASTA — GC Content & Restriction Sites**
```bash
python3 main.py sequences.fasta "Calculate GC content for each sequence and find EcoRI restriction sites in sequences with GC > 60%"
```

**2. VCF — Variant Filtering**
```bash
python3 main.py variants.vcf "Filter to keep only SNPs with quality > 30 and depth > 10, then summarize variant type distribution"
```

**3. CSV — Differential Expression**
```bash
python3 main.py expression_counts.csv "Normalize using TPM, find genes with 2-fold difference between treatment and control, plot heatmap of top 5"
```

### Logging

All stdout is duplicated to `output.log` via a custom `Logger` class. This captures the full session including the Writer's blueprint, the Coder's generated script, and the sandbox execution results — useful for debugging when the generated code fails.

### Known Limitations

- **Token limit**: Model output is capped at 2048 tokens (~100-150 lines of code). Very complex multi-step analyses may get truncated.
- **No retry loop**: If the generated code fails in the sandbox, the error is printed but the system does not currently re-attempt with the error context.
- **Single file input**: Only one data file can be provided per run.
- **No GPU, no run**: The model requires CUDA. There is no CPU fallback.
- **Generated code is unsandboxed within the temp directory**: The code can read/write files and make network calls inside the sandbox folder. It cannot escape the temp directory, but review `solution.py` before using results in production.
