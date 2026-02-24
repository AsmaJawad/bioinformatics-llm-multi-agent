from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

#Domain constrains to force LLM to accurately identify the problem scope 
#and effectively communicate it to the coder LLM

#Biological entities
class bio_entity(str, Enum):
    DNA = "DNA"
    RNA = "RNA"
    PROTEIN = "Protein"
    CELL_TYPE = "CELL_TYPE"
    ORGANISM_NAME = "ORGANISM_NAME"

#Expected file formats
class file_format(str, Enum):
    FASTA = "FASTA"
    FASTQ = "FASTQ"
    VCF = "VCF"
    LONGFORM = "LONG_FORMAT" #row has one or more "identifier" columns, and one or more "value" columns
    MATRIX = "MATRIX" #rows have id, columns have id, it's a data matrix

#determine the identifier type from given file and ensure accurate ID translation
class identifier_type(BaseModel):
    """Detects identifier provided in data file"""

    detected_type: str = Field(..., description="Type of identifier found in the file e.g. gene name, gene ID, protein ID")
    example_id: str = Field(..., description="A specific identifier found in the file e.g. gene ID = GENE001")
    db_source: str = Field(..., description="The database this ID belongs to e.g. HGNC")

    #handle potential identifier translation issues
    requires_translation: bool = Field(default=False, description="True if coder needs ID converted to a different type e.g. gene name to gene_id")
    target_id: Optional[str] = Field(None, description="If translated, what is the target identifier type?")

#using pydantic's base model ensures that LLMs execute instructions according to the generated JSON file
#forces writer LLM to think out loud before answering -> significantly reduces hallucination errors
class execution_step(BaseModel):
    step_id: int
    task_name: str
    action_desc: str = Field(..., description="Detailed technical instruction for the coder")
    tools_req: List[str] = Field(default_factory=list, json_schema_extra={"example": ["Biopython", "pandas", "bcftools"]})

class math_transformations(str, Enum):
    NORMALIZATION = "normalization"
    LOG_TRANSFORMATION = "log_transformation"
    FILTERING = "filtering"
    STANDARDIZATION = "standardization"
    LIGNMENT = "alignment"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOVA = "anova"
    PCA = "pca"
    CUSTOM = "custom" # for other transformations that are not listed

#final outputted JSON file that the writer passes to the coder for detailed instructions
class sys_blueprint(BaseModel):
    """The output of the writer Agent."""
    
    chain_of_thought: str = Field(..., description="Identify biological problem and generate step-by-step action list in plain English.")
    biological_entity: bio_entity
    primary_format: file_format
    identifier: identifier_type
    transformations: List[math_transformations]
    execution_plan: List[execution_step]
    correct_output: str = Field(..., description="What should the program print for bioinformatic analysis")

class final_result(BaseModel):
    """The results captured from bash sandbox"""
    success: bool
    stdout: str
    stderr: Optional[str] = None
    exit_code: int
    temp_dir_path: str
