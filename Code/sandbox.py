import subprocess
import os
import tempfile
import shutil

def execute_in_sandbox(bash_code, target_file):
    """
    Executes code in a truly isolated, temporary directory that 
    is destroyed immediately after execution.
    """
    # 1. Create a unique temporary directory
    # This prevents different runs from seeing each other's venvs
    temp_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix="sandbox_")
    script_path = os.path.join(temp_dir, "run_analysis.sh")
    try:
        if os.path.exists(target_file):
                shutil.copy(target_file, temp_dir)
        else:
            return f"ORCHESTRATOR ERROR: {target_file} not found in main directory."
        
        # Copy your data file into the sandbox so the script can see it
        # If your file is 'variants.vcf' or 'expression_counts.csv'
        # shutil.copy("your_data_file.vcf", temp_dir) 

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

        # Copy generated solution.py back to main directory if it exists
        solution_path = os.path.join(temp_dir, "solution.py")
        if os.path.exists(solution_path):
            shutil.copy(solution_path, os.path.join(os.getcwd(), "solution.py"))

        # Copy any output files (e.g., filtered VCFs, PNGs) back
        for f in os.listdir(temp_dir):
            full = os.path.join(temp_dir, f)
            if os.path.isfile(full) and f not in ("run_analysis.sh", os.path.basename(target_file)):
                dest = os.path.join(os.getcwd(), f)
                if not os.path.exists(dest):
                    shutil.copy(full, dest)

        return result.stdout

    except subprocess.CalledProcessError as e:
        return f"SANDBOX ERROR:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    finally:
        # 3. THE DESTRUCTOR: This wipes the venv, the script, and the temp folder
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)