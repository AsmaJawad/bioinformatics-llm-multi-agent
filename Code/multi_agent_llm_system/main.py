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
import threading
import time
import itertools

# =====================================================================
# GLOBAL FLAGS
# =====================================================================
VERBOSE = False  # Set by -T flag

# =====================================================================
# LOGGING — always writes to output.log, terminal output controlled by VERBOSE
# =====================================================================
class Logger(object):
    def __init__(self, verbose=False):
        self.terminal = sys.__stdout__
        self.log = open("output.log", "a")
        self.verbose = verbose

    def write(self, message):
        # Always write to log file
        self.log.write(message)
        self.log.flush()
        # Only write to terminal if verbose mode
        if self.verbose:
            self.terminal.write(message)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

# =====================================================================
# THINKING ANIMATION
# =====================================================================
class Spinner:
    """Animated thinking spinner, like Claude's UI."""

    PHASES = [
        ("dots",    ["   ", ".  ", ".. ", "..."]),
        ("braille", ["\u2801", "\u2803", "\u2807", "\u280f", "\u281f", "\u283f", "\u287f", "\u28ff",
                     "\u28fe", "\u28fc", "\u28f8", "\u28f0", "\u28e0", "\u28c0", "\u2880", "\u2800"]),
        ("bounce",  ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588",
                     "\u2587", "\u2586", "\u2585", "\u2584", "\u2583", "\u2582"]),
    ]

    COLORS = {
        "cyan":    "\033[96m",
        "yellow":  "\033[93m",
        "green":   "\033[92m",
        "magenta": "\033[95m",
        "reset":   "\033[0m",
        "dim":     "\033[2m",
        "bold":    "\033[1m",
    }

    def __init__(self, message="Thinking", color="cyan"):
        self.message = message
        self.color = self.COLORS.get(color, self.COLORS["cyan"])
        self.reset = self.COLORS["reset"]
        self.dim = self.COLORS["dim"]
        self.bold = self.COLORS["bold"]
        self._stop = threading.Event()
        self._thread = None
        self.start_time = None

    def _animate(self):
        terminal = sys.__stdout__
        self.start_time = time.time()

        # Cycle through animation phases for variety
        all_frames = []
        for _, frames in self.PHASES:
            all_frames.extend(frames * 3)  # repeat each phase a few times
        frame_cycle = itertools.cycle(all_frames)

        while not self._stop.is_set():
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"

            frame = next(frame_cycle)
            line = f"\r  {self.color}{frame}{self.reset} {self.bold}{self.message}{self.reset} {self.dim}({time_str}){self.reset}  "
            terminal.write(line)
            terminal.flush()
            self._stop.wait(0.1)

        # Clear the spinner line
        terminal.write("\r" + " " * 60 + "\r")
        terminal.flush()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, final_message=None):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if final_message:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"
            terminal = sys.__stdout__
            terminal.write(f"  {self.COLORS['green']}\u2714{self.COLORS['reset']} {final_message} {self.dim}({time_str}){self.COLORS['reset']}\n")
            terminal.flush()

def user_print(message):
    """Always prints to terminal regardless of verbose mode."""
    sys.__stdout__.write(message + "\n")
    sys.__stdout__.flush()

def user_input(prompt=""):
    """input() that always shows the prompt on terminal regardless of Logger."""
    sys.__stdout__.write(prompt)
    sys.__stdout__.flush()
    return input()

# =====================================================================
# MODEL LOADING
# =====================================================================
model_id = "hemanthkari/llama-3.1-pro-coder-v1"
tokenizer = None
model = None

def load_model():
    """Load the model and tokenizer. Call this once before using call_llama."""
    global tokenizer, model
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

# Standard library modules that should never be pip installed
STDLIB = {
    "sys", "os", "re", "math", "json", "csv", "collections", "itertools",
    "functools", "operator", "string", "io", "pathlib", "glob", "shutil",
    "tempfile", "subprocess", "threading", "multiprocessing", "time",
    "datetime", "copy", "random", "statistics", "typing", "abc",
    "dataclasses", "enum", "struct", "hashlib", "hmac", "secrets",
    "urllib", "http", "socket", "email", "html", "xml", "logging",
    "unittest", "pprint", "textwrap", "difflib", "contextlib",
}

def clean_stdlib_installs(script):
    """Remove standard library modules from pip install lines."""
    cleaned = []
    for line in script.split("\n"):
        if line.strip().startswith("pip install"):
            pkgs = line.strip().split()[2:]
            filtered = [p for p in pkgs if p.lstrip("-") not in STDLIB]
            if filtered:
                cleaned.append(f"pip install {' '.join(filtered)}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)

def run_pipeline(user_query, target_file):

    try:
        with open(target_file, 'r') as f:
            snippet = f.read(1000)
        print(f"Successfully read snippet from {target_file}")
    except FileNotFoundError:
        user_print(f"ERROR: '{target_file}' not found in the current folder!")
        return

    # --- Phase 1: Writer (Architect) ---
    spinner = Spinner("Analyzing your data and building execution plan", color="cyan")
    spinner.start()

    writer_config = get_writer_config()
    writer_input = format_writer_user_prompt(user_query, snippet)
    blueprint_json = call_llama(writer_config, writer_input)

    spinner.stop("Blueprint generated")

    print(f"\n--- Architect Blueprint ---\n{blueprint_json}\n")

    # --- Phase 2: Coder (Programmer) ---
    spinner = Spinner("Writing analysis code", color="yellow")
    spinner.start()

    coder_config = get_coder_config()

    coder_input = (
    f"Write a self-contained bash script to implement the following blueprint.\n\n"
    f"RULES:\n"
    f"1. SETUP: pip install every THIRD-PARTY library your Python code imports BEFORE running python. "
    f"Do NOT install standard library modules (sys, os, collections, re, math, json, csv, itertools, etc.).\n"
    f"2. FILENAME: Use the literal string '{target_file}' in your Python code. "
    f"Do NOT use variables — use the filename directly.\n"
    f"3. OUTPUT: Print ALL results to stdout using print(). Do NOT write to files.\n"
    f"4. FORMAT: Parse according to the blueprint's 'primary_format'.\n"
    f"5. LOGIC: Follow the 'execution_plan' steps exactly.\n\n"
    f"You MUST follow this EXACT bash script structure:\n"
    f"```\n"
    f"#!/bin/bash\n"
    f"set -e\n"
    f"python3 -m venv venv\n"
    f"source venv/bin/activate\n"
    f"pip install <libraries>\n"
    f"cat << 'EOF' > solution.py\n"
    f"<your python code here — use print() for output>\n"
    f"EOF\n"
    f"python3 solution.py\n"
    f"deactivate\n"
    f"```\n\n"
    f"CRITICAL: You MUST include the EOF line by itself to close the heredoc, "
    f"then python3 solution.py MUST come AFTER the EOF line.\n\n"
    f"Output ONLY the bash script, no explanations.\n\n"
    f"BLUEPRINT:\n{blueprint_json}"
    )

    bash_script = call_llama(coder_config, coder_input)
    bash_script = bash_script.replace("```bash", "").replace("```", "").strip()

    bash_script = clean_stdlib_installs(bash_script)

    spinner.stop("Code generated")

    print(f"\n--- Generated Script ---\n{bash_script}\n")

    # --- Phase 3: Sandbox Execution with Self-Correction ---
    MAX_RETRIES = 3
    attempt = 1

    while attempt <= MAX_RETRIES:
        spinner = Spinner(f"Running code (attempt {attempt}/{MAX_RETRIES})", color="green")
        spinner.start()

        result = execute_in_sandbox(bash_script, target_file)
        print(result)

        if not result.startswith("SANDBOX ERROR"):
            spinner.stop("Execution succeeded")
            break

        spinner.stop(f"Attempt {attempt} failed")

        if attempt < MAX_RETRIES:
            user_print(f"\n\033[93mAttempt {attempt} failed — asking AI to fix the code...\033[0m")

            # Feed the error + script back to the Coder to self-correct
            fix_input = (
                f"The following bash script failed when executed.\n\n"
                f"SCRIPT:\n{bash_script}\n\n"
                f"ERROR:\n{result}\n\n"
                f"Fix the script so it runs without errors. "
                f"The data file is '{target_file}'. "
                f"Do NOT install standard library modules (sys, os, collections, re, math, etc.). "
                f"Use print() for all output. "
                f"Keep the same structure: venv setup, pip install, cat << 'EOF' > solution.py, EOF, python3 solution.py.\n"
                f"Output ONLY the corrected bash script, no explanations."
            )

            spinner = Spinner(f"Rewriting code (attempt {attempt + 1})", color="yellow")
            spinner.start()

            bash_script = call_llama(coder_config, fix_input)
            bash_script = bash_script.replace("```bash", "").replace("```", "").strip()
            bash_script = clean_stdlib_installs(bash_script)

            spinner.stop("Revised code generated")
            print(f"\n--- Revised Script (attempt {attempt + 1}) ---\n{bash_script}\n")

        attempt += 1

    # --- Final Output (always shown) ---
    user_print("\n" + "=" * 50)
    user_print("  RESULTS")
    user_print("=" * 50)

    if result.startswith("SANDBOX ERROR"):
        user_print(f"\n\033[91mExecution failed after {MAX_RETRIES} attempts.\033[0m")
        user_print(result)
        user_print(f"\nFull logs saved to: output.log")
    else:
        user_print(f"\n{result}")
        if os.path.exists("solution.py"):
            user_print(f"\033[92mGenerated script saved to: solution.py\033[0m")
        user_print(f"\033[2mFull logs saved to: output.log\033[0m")


def detect_format(filename):
    """Detect file format from extension."""
    ext = os.path.splitext(filename)[1].lower()
    formats = {
        '.fasta': 'FASTA', '.fa': 'FASTA', '.fna': 'FASTA',
        '.vcf': 'VCF',
        '.csv': 'CSV', '.tsv': 'TSV',
    }
    return formats.get(ext, 'Unknown')

def interactive_prompt(target_file):
    """Guide the user through building a query with clarifying questions."""
    file_format = detect_format(target_file)
    user_print(f"\nDetected file format: {file_format}")
    user_print(f"File: {target_file}")
    user_print("=" * 50)

    # Question 1: What type of analysis?
    user_print("\nWhat type of analysis would you like to perform?")
    if file_format == 'FASTA':
        options = [
            "GC content analysis & restriction site detection",
            "Sequence length statistics",
            "Custom (describe your own)"
        ]
    elif file_format == 'VCF':
        options = [
            "Filter variants by quality and depth",
            "Variant type distribution summary",
            "Custom (describe your own)"
        ]
    elif file_format in ('CSV', 'TSV'):
        options = [
            "Gene expression normalization (TPM) & differential analysis",
            "Statistical summary of columns",
            "Custom (describe your own)"
        ]
    else:
        options = ["Custom (describe your own)"]

    for i, opt in enumerate(options, 1):
        user_print(f"  [{i}] {opt}")

    choice = user_input("\nSelect an option (number): ").strip()

    # If they picked a preset, build the query automatically
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(options) - 1:
            base_query = options[choice_idx]
        else:
            base_query = None
    except ValueError:
        base_query = None

    if base_query is None:
        base_query = user_input("\nDescribe what you want to do with the data:\n> ").strip()

    # Question 2: Any filters or thresholds?
    user_print("\nDo you have any specific thresholds or filters?")
    user_print("  [1] Use defaults")
    user_print("  [2] Specify my own")
    filter_choice = user_input("Select (1 or 2): ").strip()

    if filter_choice == "2":
        filters = user_input("Describe your thresholds (e.g., 'quality > 30, depth > 10'):\n> ").strip()
        base_query += f". Apply these filters: {filters}"

    # Question 3: Output preference
    user_print("\nHow would you like the output?")
    user_print("  [1] Print results to terminal")
    user_print("  [2] Save to an output file")
    user_print("  [3] Both")
    out_choice = user_input("Select (1, 2, or 3): ").strip()

    if out_choice == "2":
        base_query += ". Save all results to an output file instead of printing."
    elif out_choice == "3":
        base_query += ". Print results to terminal AND save them to an output file."

    # Build the final query
    final_query = f"{base_query} The input file is {target_file}."
    user_print(f"\n{'=' * 50}")
    user_print(f"Final query: {final_query}")
    user_print(f"{'=' * 50}\n")

    return final_query

if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse -T flag
    if "-T" in args:
        VERBOSE = True
        args.remove("-T")

    if len(args) < 1:
        user_print("Usage: python3 main.py [-T] <data_file> [query]")
        user_print("")
        user_print("Options:")
        user_print("  -T          Testing mode: show full logs (blueprint, script, sandbox output)")
        user_print("")
        user_print("Examples:")
        user_print("  python3 main.py sequences.fasta                     # Interactive mode")
        user_print("  python3 main.py -T variants.vcf                     # Interactive + verbose logging")
        user_print('  python3 main.py expression_counts.csv "Normalize with TPM"  # Direct query')
        sys.exit(1)

    target_file = args[0]

    if not os.path.exists(target_file):
        user_print(f"ERROR: File '{target_file}' not found!")
        sys.exit(1)

    # Banner
    user_print("")
    user_print("\033[96m" + "=" * 50)
    user_print("  BioAgent - Multi-Agent LLM Analysis System")
    user_print("=" * 50 + "\033[0m")
    if VERBOSE:
        user_print("\033[93m  [TESTING MODE] Full logging enabled\033[0m")
    user_print("")

    # Load model (output goes to terminal normally)
    load_model()

    # Set up logger after model loads
    sys.stdout = Logger(verbose=VERBOSE)

    # If a query is provided as remaining args, use it directly
    if len(args) > 1:
        USER_QUERY = " ".join(args[1:])
        user_print(f"Query: {USER_QUERY}")
    else:
        # Interactive mode
        USER_QUERY = interactive_prompt(target_file)

    run_pipeline(USER_QUERY, target_file)