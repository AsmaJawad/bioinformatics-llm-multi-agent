import subprocess
import os
import tempfile
import shutil

def execute_in_sandbox(bash_code, target_file, output_dir="output"):
    """
    Executes code in a truly isolated, temporary directory that
    is destroyed immediately after execution.
    Results are copied to output_dir.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create a unique temporary directory
    # This prevents different runs from seeing each other's venvs
    temp_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix="sandbox_")
    script_path = os.path.join(temp_dir, "run_analysis.sh")
    try:
        if os.path.exists(target_file):
            # Preserve the relative path structure inside the sandbox so that
            # code referencing 'input/file.csv' resolves correctly.
            dest_path = os.path.join(temp_dir, target_file)
            os.makedirs(os.path.dirname(dest_path) or temp_dir, exist_ok=True)
            shutil.copy(target_file, dest_path)
            # Also drop a copy at the sandbox root so bare-basename references work too.
            root_copy = os.path.join(temp_dir, os.path.basename(target_file))
            if not os.path.exists(root_copy):
                shutil.copy(target_file, root_copy)
        else:
            return f"ORCHESTRATOR ERROR: {target_file} not found in main directory."

        with open(script_path, "w") as f:
            f.write(bash_code)
        
        os.chmod(script_path, 0o755)
    
        result = subprocess.run(
                ["bash", "run_analysis.sh"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True
            )

        # Copy generated solution.py back to output directory if it exists
        solution_path = os.path.join(temp_dir, "solution.py")
        if os.path.exists(solution_path):
            shutil.copy(solution_path, os.path.join(output_dir, "solution.py"))

        # Copy any output files (e.g., filtered VCFs, PNGs) back to output dir
        for f in os.listdir(temp_dir):
            full = os.path.join(temp_dir, f)
            if os.path.isfile(full) and f not in ("run_analysis.sh", os.path.basename(target_file)):
                dest = os.path.join(output_dir, f)
                if not os.path.exists(dest):
                    shutil.copy(full, dest)

        return result.stdout

    except subprocess.CalledProcessError as e:
        return f"SANDBOX ERROR:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    finally:
        # 3. THE DESTRUCTOR: This wipes the venv, the script, and the temp folder
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)