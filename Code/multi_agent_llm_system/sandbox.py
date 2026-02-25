#manages the isolated execution environment. 
# It creates a temporary directory, 
# sets up a temporary virtual environment, 
# and runs the Coder's generated script safely away from your main files.

import subprocess
import os

def execute_in_sandbox(bash_code):
    """
    Executes the Cored-generated Bash script and returns the stdout.
    """
    script_name = "run_analysis.sh"
    
    with open(script_name, "w") as f:
        f.write(bash_code)
    
    # Make script executable
    os.chmod(script_name, 0o755)
    
    try:
        # Execute and capture output
        result = subprocess.run(
            ["bash", script_name],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"SANDBOX ERROR:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    finally:
        # Clean up the entry script
        if os.path.exists(script_name):
            os.remove(script_name)