from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

#Domain constrains to force LLM to choose valid categories for accurate reading of data
class bio_category(str, Enum):
    DNA = "DNA"
    RNA = "RNA"
    PROTEIN = "Protein"
    METAGENOMIC = "Metagenomic"
    CLINICAL = "Clinical"

#Expected file formats
class file_format(str, Enum):
    FASTA = "FASTA"
    FASTQ = "FASTQ"
    VCF = "VCF"
    BAM = "BAM"
    CSV = "CSV/TSV"
    BED = "BED"

class execution_step(BaseModel):
    step_id: int
    task_name: str
    action_desc: str = Field(..., description="Detailed technical instruction for the coder")
    tools_req: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Biopython", "pandas", "bcftools"]})

class sys_blueprint(BaseModel):
    """The output of the writer Agent."""
    chain_of_thought: str = Field(..., description="Identify biological problem and generate step-by-step action list in plain English.")

    biological_entity: bio_category
    primary_format: file_format

    execution_plan: List[execution_step]

    correct_output: str = Field(..., description="What should the program print for bioinformatic analysis")

class final_result(BaseModel):
    """The results captured from bash sandbox"""
    success: bool
    stdout: str
    stderr: Optional[str] = None
    exit_code: int
    temp_dir_path: str
