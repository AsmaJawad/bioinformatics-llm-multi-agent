# pip install git+https://github.com/huggingface/transformers.git # TODO: merge PR to main

# ==================== IMPORTS ====================
# AutoModelForCausalLM: Loads a causal language model (text generation model) from Hugging Face
# AutoTokenizer: Loads the tokenizer that converts text to tokens the model understands
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch: PyTorch library used for tensor operations and GPU acceleration
import torch

# re: Regular expressions library used to pattern match and clean up the AI's response text
import re

# traceback: Provides detailed error stack traces so the AI gets full context when fixing errors
import traceback

# signal: Used to set a timer (alarm) that kills code execution if it runs too long (e.g., infinite loops)
import signal


# ==================== MODEL LOADING ====================
# This is the Hugging Face model ID for the Llama 3.1 coding-specialized model.
# It gets downloaded from Hugging Face Hub on first run and cached locally for future runs.
model_id = "hemanthkari/llama-3.1-pro-coder-v1"

# Load the tokenizer for this model.
# The tokenizer converts human-readable text into numerical token IDs that the model can process,
# and converts the model's output token IDs back into human-readable text.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the actual language model onto the GPU.
# torch_dtype=torch.bfloat16: Uses half-precision floating point to use less GPU memory
#   (bfloat16 uses 16 bits instead of 32 bits per number, cutting memory usage roughly in half).
# device_map="auto": Automatically places the model on available GPUs (or CPU if no GPU).
#   If the model is too large for one GPU, it will split across multiple GPUs automatically.
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)


# ==================== AI PROMPT FUNCTION ====================
def getCoderAI(usersPrompt):
    """
    Takes a user's text prompt and sends it to the Llama model to generate Python code.

    How it works:
    1. Wraps the user's prompt in a chat message format with a system instruction
    2. Tokenizes the messages into numerical IDs the model understands
    3. Runs the model to generate new tokens (the AI's response)
    4. Decodes those tokens back into readable text and returns it

    Args:
        usersPrompt (str): The user's description of what they want the Python script to do.

    Returns:
        str: The raw text response from the AI model (should be Python code).
    """

    prompt = usersPrompt

    # Build the chat messages in the format the model expects.
    # "system" role: Tells the AI how to behave and what rules to follow.
    # "user" role: The actual request from the user.
    messages = [
        {
            "role": "system",
            "content": (
                # Instruct the AI to only output code, no explanations or markdown formatting
                "You are an expert Python developer. Output ONLY valid Python code. No explanations, no markdown.\n"
                # The code should be a full script that runs on its own, not just a snippet
                "Always produce a complete, self-contained Python script that can be run directly.\n"
                # Make sure imports like 'import math' are included so the code doesn't fail
                "Include all necessary imports at the top.\n"
                # Define any helper functions or classes the code needs
                "Define any needed functions or classes.\n"
                # Always include a main() function so we can actually see the code run and produce output
                "Always include a main() function that demonstrates the code by calling the functions with example inputs and printing the results.\n"
                # Include the standard Python entry point guard so main() actually gets called
                "Always include an `if __name__ == '__main__': main()` block at the end.\n"
                # If the user just says "sort a list" without giving specific data, the AI should
                # make up its own example data so the code actually does something when executed
                "If the user does not provide specific test cases or examples, create your own realistic example inputs to demonstrate the code works.\n"
                # Prevent the AI from generating huge example data that exceeds the token limit
                # and gets cut off mid-line, causing unterminated string literals
                "IMPORTANT: Keep all example data SHORT and concise. Never generate long strings or large datasets as examples. Use small, brief examples."
            ),
        },
        {
            "role": "user",
            # Embed the user's prompt into a clear instruction for the AI
            "content": f"Write a complete runnable Python script for: {prompt}"
        },
    ]

    # Convert the chat messages into token IDs using the model's chat template.
    # return_tensors="pt": Return as PyTorch tensors (required for the model).
    # add_generation_prompt=True: Adds the special token that tells the model
    #   "now it's your turn to respond" so it starts generating.
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )

    # Move the input tokens to the same device (GPU/CPU) where the model is loaded.
    # If the model is on GPU, the inputs need to be on GPU too, otherwise PyTorch errors out.
    inputs = inputs.to(model.device)

    # Create an attention mask — a tensor of 1s the same shape as inputs.
    # This tells the model to pay attention to ALL input tokens (none are padding).
    # Without this, the model prints a warning because the pad token and end-of-sequence
    # token are the same, and it can't tell which tokens are real vs padding.
    attention_mask = torch.ones_like(inputs)

    # Run the model to generate new tokens (the AI's code response).
    # max_new_tokens=1024: Generate at most 1024 new tokens (increased from 512 to avoid
    #   the response getting cut off mid-line, which causes unterminated string literals).
    # do_sample=False: Use greedy decoding — always pick the most likely next token.
    #   This makes output deterministic (same input = same output every time).
    # pad_token_id=tokenizer.eos_token_id: Tells the model what token to use for padding.
    #   Set to end-of-sequence token since this model doesn't have a dedicated pad token.
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated tokens back into human-readable text.
    # outputs[0]: Get the first (and only) sequence from the batch.
    # [inputs.shape[1]:]: Skip over the input tokens — we only want the NEW tokens
    #   the model generated (the AI's response), not the original prompt echoed back.
    # skip_special_tokens=True: Remove special tokens like <|end_of_text|> from output.
    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    return response


# ==================== EXECUTION TIMEOUT ====================
class ExecutionTimeout(Exception):
    """
    Custom exception class that gets raised when AI-generated code takes too long to execute.
    This prevents infinite loops or extremely slow code from hanging the program forever.
    Inherits from Exception so it can be caught in try/except blocks.
    """
    pass


def _timeout_handler(signum, frame):
    """
    Signal handler function that gets called when the SIGALRM timer goes off.
    This is triggered by the operating system after the timeout period expires.

    Args:
        signum: The signal number (will be SIGALRM). Required by signal handler interface.
        frame: The current stack frame when the signal was received. Required by signal handler interface.

    Raises:
        ExecutionTimeout: Always raises this to interrupt whatever code is currently running.
    """
    raise ExecutionTimeout("Code execution timed out (30 second limit)")


# ==================== CODE CLEANING ====================
def clean_code_response(raw_response):
    """
    Takes the raw text output from the AI model and extracts only the valid Python code.

    The AI sometimes wraps code in markdown fences (```python ... ```) or adds
    explanatory text before/after the code. This function strips all of that away
    so we're left with just executable Python code.

    Args:
        raw_response (str): The raw text response from the AI model.

    Returns:
        str: Cleaned Python code ready to be executed.
    """
    # Remove leading/trailing whitespace from the entire response
    code = raw_response.strip()

    # Remove markdown code fences that the AI might wrap around the code.
    # Matches ``` or ```python at the start, and ``` at the end.
    # The AI is told not to use markdown, but sometimes it does anyway.
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)

    # Split the code into individual lines so we can analyze each one
    lines = code.splitlines()

    # ---- Find where the actual Python code starts ----
    # Sometimes the AI puts text like "Here's the code:" before the actual code.
    # We scan through lines looking for the first line that looks like real Python:
    #   - import/from statements (e.g., "import math")
    #   - function/class definitions (e.g., "def my_func():")
    #   - comments (e.g., "# This function...")
    #   - if statements (e.g., "if __name__...")
    #   - docstrings (""" or ''')
    #   - variable assignments (e.g., "x = 5")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (
            stripped.startswith(("import ", "from ", "def ", "class ", "#", "if ", "\"\"\"", "'''"))
            or re.match(r"^[a-zA-Z_]\w*\s*=", stripped)
        ):
            start_idx = i
            break

    # ---- Find where the actual Python code ends ----
    # Sometimes the AI adds explanatory text after the code like "Note: this function..."
    # We scan backwards from the end looking for the last line that ISN'T an explanation.
    # Lines starting with "Note:", "Output:", "Explanation:", or "This " are considered
    # non-code text that should be removed.
    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith(("Note:", "Output:", "Explanation:", "This ")):
            end_idx = i + 1
            break

    # Join only the lines between start and end — this is the clean code
    code = "\n".join(lines[start_idx:end_idx])

    # If after all that cleaning we ended up with nothing, fall back to
    # the original response (better to try running something than nothing)
    if not code.strip():
        code = raw_response.strip()

    return code


# ==================== SYNTAX VALIDATION ====================
def validate_syntax(code):
    """
    Checks if the given code string is valid Python syntax WITHOUT actually running it.

    Uses Python's built-in compile() function which parses the code and checks for
    syntax errors (missing colons, unmatched parentheses, invalid indentation, etc.)
    but does NOT execute the code — so it's safe to call on untrusted code.

    This is useful because syntax errors are easy to detect early, and we can send
    the specific syntax error message to the AI for a more targeted fix.

    Args:
        code (str): The Python code string to validate.

    Returns:
        tuple: (True, None) if syntax is valid, or (False, SyntaxError) if invalid.
    """
    try:
        # compile() parses the code into a code object.
        # "<ai_generated>": A label for error messages so we know where the code came from.
        # "exec": Compilation mode for a sequence of statements (as opposed to a single expression).
        compile(code, "<ai_generated>", "exec")
        return True, None
    except SyntaxError as e:
        # Syntax is invalid — return the SyntaxError object which contains
        # the line number, offset, and description of what went wrong.
        return False, e


# ==================== SAFE CODE EXECUTION ====================
def safe_exec(code, timeout=30):
    """
    Safely executes AI-generated Python code with multiple layers of protection:

    1. SYNTAX CHECK: Validates syntax before running (catches errors early)
    2. TIMEOUT: Sets a 30-second alarm so infinite loops don't hang forever
    3. ISOLATION: Runs in a separate globals dictionary so it can't mess with our variables
    4. OUTPUT CAPTURE: Captures both stdout and stderr so we can see what the code printed
    5. ERROR CATCHING: Catches ALL possible exceptions including SystemExit, KeyboardInterrupt,
       and any other BaseException subclass

    Args:
        code (str): The Python code string to execute.
        timeout (int): Maximum seconds the code is allowed to run. Default 30 seconds.

    Returns:
        tuple: (success: bool, result: str)
            - If success is True, result contains the captured stdout/stderr output.
            - If success is False, result contains the error message and full traceback.
    """
    # io: Used to create in-memory string buffers to capture print output
    # contextlib: Used to redirect stdout/stderr to our buffers
    import io
    import contextlib

    # ---- Step 1: Validate syntax before running ----
    # No point trying to execute code that has syntax errors.
    # This also gives us a cleaner error message for the AI to fix.
    valid, syntax_err = validate_syntax(code)
    if not valid:
        return False, f"SyntaxError: {syntax_err}"

    # ---- Step 2: Set up a timeout alarm ----
    # SIGALRM is a Unix/Linux signal that fires after a set number of seconds.
    # We save the old handler so we can restore it when we're done.
    # If SIGALRM isn't available (Windows), we skip the timeout — the code
    # could theoretically hang, but this is better than crashing.
    old_handler = None
    try:
        # Save whatever signal handler was there before (so we can restore it later)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        # Start a countdown timer — after `timeout` seconds, SIGALRM fires
        # and _timeout_handler raises ExecutionTimeout
        signal.alarm(timeout)
    except (OSError, AttributeError):
        # SIGALRM not available on this OS (Windows) — skip timeout protection
        pass

    try:
        # ---- Step 3: Set up isolated execution environment ----
        # Create a fresh globals dictionary for the exec'd code.
        # __builtins__ is included so the code can use built-in functions like print(), len(), etc.
        # This isolation means the AI's code can't accidentally read or overwrite
        # our variables (like `model`, `tokenizer`, etc.)
        # Set __name__ to "__main__" so that the AI's `if __name__ == '__main__': main()`
        # block actually triggers. Without this, __name__ would be undefined or wrong
        # inside exec(), and the AI's main() function would never get called.
        ai_globals = {"__builtins__": __builtins__, "__name__": "__main__"}

        # Create in-memory string buffers to capture everything the code prints.
        # stdout_capture: Catches regular print() output
        # stderr_capture: Catches error/warning output written to stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # ---- Step 4: Execute the code with output redirection ----
        # redirect_stdout/redirect_stderr temporarily reroute all print output
        # into our string buffers instead of showing on the terminal.
        # This lets us collect the output and display it in a controlled way.
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, ai_globals)

        # ---- Step 5: Collect the captured output ----
        # Get everything that was printed to stdout
        output = stdout_capture.getvalue()
        # Get everything that was printed to stderr (warnings, error messages)
        err_output = stderr_capture.getvalue()
        # If there was stderr output, append it to the main output with a label
        if err_output:
            output += f"\n[stderr]: {err_output}"

        # Code ran successfully — return True with the output
        return True, output

    # ---- Exception Handlers (ordered from most specific to most general) ----

    except ExecutionTimeout as e:
        # The 30-second alarm went off — code was taking too long (likely an infinite loop)
        return False, str(e)

    except SystemExit as e:
        # The AI's code called sys.exit() — this would normally kill our entire program.
        # We catch it here so it only stops the AI's code, not our main loop.
        # We treat this as "success" since the code ran, it just chose to exit.
        return True, f"(Code called sys.exit with code: {e.code})"

    except KeyboardInterrupt:
        # The user pressed Ctrl+C while the AI's code was running.
        # We catch this to report it cleanly instead of crashing.
        return False, "Code execution was interrupted by user"

    except BaseException as e:
        # Catch absolutely EVERYTHING else — this is the nuclear option.
        # BaseException is the parent of ALL Python exceptions including:
        #   - Exception (and all its subclasses: TypeError, ValueError, NameError, etc.)
        #   - GeneratorExit, SystemExit, KeyboardInterrupt (already caught above)
        # We use traceback.format_exc() to get the full stack trace, which shows
        # exactly which line caused the error — this helps the AI fix it.
        tb = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\n\nFull traceback:\n{tb}"

    finally:
        # ---- Cleanup: Always cancel the timeout alarm ----
        # This runs no matter what happened above (success, error, timeout, etc.)
        # signal.alarm(0) cancels any pending alarm so it doesn't fire later
        # and crash something else. We also restore the original signal handler.
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (OSError, AttributeError):
            pass


# ==================== MAIN LOOP ====================
def main():
    """
    The main interactive loop that:
    1. Prompts the user for a description of what they want
    2. Sends it to the AI to generate Python code
    3. Cleans and executes the generated code
    4. If the code has errors, sends the error back to the AI to fix (up to 3 retries)
    5. Repeats until the user types "Exit"
    """

    # Infinite loop — keeps asking for prompts until the user types "Exit" or Ctrl+C
    while 1:

        # ---- Get user input with error protection ----
        # Wrap input() in try/except because:
        # - EOFError: Happens if input stream closes (e.g., piped input runs out)
        # - KeyboardInterrupt: Happens if user presses Ctrl+C while typing
        # In either case, we exit the loop gracefully instead of crashing.
        try:
            userPrompt = input("Enter a prompt (describe what you want the Python script to do)\n Type `Exit` to quit\n")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        # ---- Check for exit command ----
        # .strip() removes whitespace so "  Exit  " also works
        if userPrompt.strip() == "Exit":
            break

        # ---- Reject empty prompts ----
        # If the user just hits Enter without typing anything, ask them to try again
        # instead of sending an empty string to the AI (which would produce garbage)
        if not userPrompt.strip():
            print("Empty prompt, please try again.")
            continue

        # ---- Send prompt to AI model ----
        # Wrap in try/except because the model could run out of GPU memory,
        # the tokenizer could fail, or any other model-related error could occur.
        # If it fails, we print the error and loop back to ask for another prompt
        # instead of crashing the whole program.
        try:
            coderAnswer = getCoderAI(userPrompt)
        except Exception as e:
            print(f"Error communicating with the AI model: {e}")
            continue

        # ---- Clean the AI's response ----
        # Strip markdown fences, explanatory text, etc. to get just the Python code
        current_code = clean_code_response(coderAnswer)

        # ---- Check for empty response ----
        # If the AI returned nothing useful (or the cleaning removed everything),
        # ask the user to try again rather than trying to execute empty code
        if not current_code.strip():
            print("AI returned an empty response. Please try again.")
            continue

        # ---- Execute with retry loop ----
        # max_retries: Maximum number of times to ask the AI to fix its own errors.
        #   After 3 failures we give up to avoid an infinite loop of bad fixes.
        # attempt: Tracks which attempt we're on (0 = first try, 1 = first retry, etc.)
        max_retries = 3
        attempt = 0

        # Keep trying until the code works or we run out of retries
        while attempt <= max_retries:

            # ---- Display the code being executed ----
            # Print with visual separators so it's easy to see what code is being run
            # and which attempt number this is
            print(f"\n{'='*60}")
            print(f"Coder Answer (attempt {attempt + 1})")
            print(f"{'='*60}")
            print(current_code)
            print(f"{'='*60}")
            print("SYSTEM OUTPUT")
            print(f"{'='*60}")

            # ---- Execute the code safely ----
            # safe_exec handles timeout, output capture, and all error catching.
            # Returns a tuple: (True/False for success, output string or error string)
            success, result = safe_exec(current_code)

            if success:
                # ---- Code ran successfully ----
                # Print whatever the code outputted (print statements, return values, etc.)
                # Then break out of the retry loop and go back to asking for a new prompt
                print(result)
                break
            else:
                # ---- Code had an error ----
                # Print the error so the user can see what went wrong
                print(f"ERROR: {result}")

                # Increment attempt counter and check if we've used all retries
                attempt += 1
                if attempt > max_retries:
                    # We've tried enough times — give up and move on to the next prompt
                    print(f"\nMax retries ({max_retries}) reached. Moving on.")
                    break

                # ---- Re-prompt the AI to fix the error ----
                print(f"\nRe-prompting AI to fix the error (attempt {attempt + 1}/{max_retries + 1})...")

                # Build a detailed fix prompt that gives the AI:
                # 1. The broken code so it knows what to fix
                # 2. The full error message and traceback so it knows what went wrong
                # 3. Explicit requirements so it doesn't forget imports or main()
                fix_prompt = (
                    f"The following Python code has an error. Fix it and return ONLY the corrected Python code.\n\n"
                    f"Original code:\n```python\n{current_code}\n```\n\n"
                    f"Error:\n{result}\n\n"
                    f"Requirements:\n"
                    f"- Fix the error\n"
                    f"- Return ONLY valid Python code, no explanations\n"
                    f"- Make sure all imports are included\n"
                    f"- Include a main() function and if __name__ == '__main__': main() block"
                )

                # ---- Get the fix from the AI ----
                # Wrap in try/except because the model could fail here too.
                # If it does, we break out of the retry loop and move on.
                try:
                    fixed_response = getCoderAI(fix_prompt)
                    # Clean the fix response the same way we cleaned the original
                    current_code = clean_code_response(fixed_response)

                    # If the AI returned nothing useful as a fix, give up
                    if not current_code.strip():
                        print("AI returned empty fix. Moving on.")
                        break
                except Exception as e:
                    # Model error during fix attempt — can't continue retrying
                    print(f"Error getting fix from AI: {e}")
                    break


# ==================== ENTRY POINT ====================
# Standard Python entry point guard.
# __name__ == "__main__" is True only when this file is run directly (e.g., python llama.py).
# It is False when this file is imported as a module by another script.
# This ensures main() only runs when we execute the file directly.
if __name__ == "__main__":
    main()
