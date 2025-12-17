"""MCP Server for chai-1

Provides both synchronous and asynchronous (submit) APIs for all tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("chai-1")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 3 min)
# ==============================================================================

@mcp.tool()
def validate_fasta_file(input_file: str) -> dict:
    """
    Validate a FASTA file format and contents for structure prediction.

    Performs quick validation of FASTA files before prediction.
    Runtime: ~1-5 seconds

    Args:
        input_file: Path to FASTA file to validate

    Returns:
        Dictionary with validation results and sequence information
    """
    try:
        # Import the validation function from our scripts
        sys.path.insert(0, str(SCRIPTS_DIR / "lib"))
        import scripts.lib.io as lib_io
        validate_fasta_file = lib_io.validate_fasta_file
        read_fasta_content = lib_io.read_fasta_content

        # Validate file exists and format
        if not Path(input_file).exists():
            return {"status": "error", "error": f"File not found: {input_file}"}

        validation_result = validate_fasta_file(input_file)
        if not validation_result:
            return {"status": "error", "error": "Invalid FASTA format"}

        # Read and analyze sequences
        sequences = read_fasta_content(input_file)
        sequence_info = []
        total_length = 0

        for header, seq in sequences:
            seq_len = len(seq)
            total_length += seq_len
            sequence_info.append({
                "header": header,
                "length": seq_len,
                "type": "protein" if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq.upper()) else "mixed"
            })

        estimated_runtime = total_length * 0.1  # rough estimate: 0.1 min per residue

        return {
            "status": "success",
            "file": input_file,
            "total_sequences": len(sequence_info),
            "total_length": total_length,
            "sequences": sequence_info,
            "estimated_runtime_minutes": round(estimated_runtime, 1),
            "recommended_api": "submit" if estimated_runtime > 3 else "sync"
        }

    except Exception as e:
        logger.error(f"FASTA validation failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def predict_small_peptide(
    sequence: str,
    max_length: int = 20,
    device: str = "cuda:0",
    output_file: Optional[str] = None
) -> dict:
    """
    Synchronous prediction for very small peptides/sequences.

    Only processes sequences up to max_length amino acids for quick results.
    Runtime: ~30 seconds to 2 minutes for very small sequences

    Args:
        sequence: Amino acid sequence (single letter code)
        max_length: Maximum sequence length to process (default: 20)
        device: Compute device (default: cuda:0)
        output_file: Optional path to save output

    Returns:
        Dictionary with prediction results or error if sequence too long
    """
    try:
        # Validate sequence length
        if len(sequence) > max_length:
            return {
                "status": "error",
                "error": f"Sequence too long ({len(sequence)} > {max_length}). Use submit_basic_prediction for longer sequences."
            }

        # Validate sequence characters
        if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence.upper()):
            return {"status": "error", "error": "Invalid amino acid characters in sequence"}

        # Import and run prediction
        from predict_basic_structure import run_predict_basic_structure
        import tempfile

        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">small_peptide\n{sequence.upper()}\n")
            temp_fasta = f.name

        try:
            # Run prediction with minimal settings for speed
            config = {
                "num_trunk_recycles": 1,  # Reduced for speed
                "num_diffn_timesteps": 50,  # Reduced for speed
                "device": device,
                "seed": 42
            }

            result = run_predict_basic_structure(
                input_file=temp_fasta,
                output_dir=Path(output_file).parent if output_file else None,
                config=config
            )

            result["status"] = "success"
            result["sequence"] = sequence
            result["sequence_length"] = len(sequence)

            return result

        finally:
            # Cleanup temp file
            Path(temp_fasta).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Small peptide prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_sequence_composition(input_file: str) -> dict:
    """
    Analyze amino acid composition and properties of sequences in a FASTA file.

    Provides quick analysis without running prediction.
    Runtime: ~1-10 seconds

    Args:
        input_file: Path to FASTA file

    Returns:
        Dictionary with composition analysis and prediction suitability
    """
    try:
        sys.path.insert(0, str(SCRIPTS_DIR / "lib"))
        import scripts.lib.io as lib_io
        validate_fasta_file = lib_io.validate_fasta_file
        read_fasta_content = lib_io.read_fasta_content

        if not Path(input_file).exists():
            return {"status": "error", "error": f"File not found: {input_file}"}

        if not validate_fasta_file(input_file):
            return {"status": "error", "error": "Invalid FASTA format"}

        sequences = read_fasta_content(input_file)
        analysis = []

        for header, seq in sequences:
            seq = seq.upper()
            seq_len = len(seq)

            # Amino acid composition
            aa_counts = {aa: seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
            aa_freq = {aa: count/seq_len for aa, count in aa_counts.items()}

            # Basic properties
            hydrophobic = sum(aa_counts[aa] for aa in "AILVFWY") / seq_len
            charged = sum(aa_counts[aa] for aa in "DEKR") / seq_len
            polar = sum(aa_counts[aa] for aa in "STNQC") / seq_len

            analysis.append({
                "header": header,
                "length": seq_len,
                "composition": aa_freq,
                "properties": {
                    "hydrophobic_fraction": round(hydrophobic, 3),
                    "charged_fraction": round(charged, 3),
                    "polar_fraction": round(polar, 3)
                },
                "prediction_complexity": "simple" if seq_len < 50 else "moderate" if seq_len < 200 else "complex"
            })

        return {
            "status": "success",
            "file": input_file,
            "sequences": analysis,
            "total_sequences": len(analysis)
        }

    except Exception as e:
        logger.error(f"Sequence analysis failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 3 min)
# ==============================================================================

@mcp.tool()
def submit_basic_prediction(
    input_file: str,
    output_dir: Optional[str] = None,
    device: str = "cuda:0",
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit basic protein structure prediction for background processing.

    This performs structure prediction from FASTA sequences using Chai-1.
    Typical runtime: 3-30+ minutes depending on sequence length and GPU.

    Args:
        input_file: Path to FASTA file with protein/DNA/RNA/ligand sequences
        output_dir: Directory to save outputs (optional)
        device: Compute device (default: cuda:0)
        num_trunk_recycles: Model recycles for accuracy (default: 3)
        num_diffn_timesteps: Diffusion timesteps (default: 200)
        seed: Random seed for reproducibility (default: 42)
        job_name: Optional name for tracking the job

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "predict_basic_structure.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "output": output_dir,
            "device": device,
            "num-trunk-recycles": num_trunk_recycles,
            "num-diffn-timesteps": num_diffn_timesteps,
            "seed": seed
        },
        job_name=job_name or "basic_prediction"
    )

@mcp.tool()
def submit_msa_prediction(
    input_file: str,
    output_dir: Optional[str] = None,
    msa_directory: Optional[str] = None,
    use_msa_server: bool = False,
    device: str = "cuda:0",
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit MSA-enhanced structure prediction for background processing.

    This performs structure prediction with Multiple Sequence Alignments
    for improved accuracy. Uses either local MSA files or online MSA server.
    Typical runtime: 5-60+ minutes (includes MSA retrieval time).

    Args:
        input_file: Path to FASTA file with protein sequences
        output_dir: Directory to save outputs (optional)
        msa_directory: Local directory with MSA files (.aligned.pqt format)
        use_msa_server: Use online MSA server instead of local files
        device: Compute device (default: cuda:0)
        num_trunk_recycles: Model recycles for accuracy (default: 3)
        num_diffn_timesteps: Diffusion timesteps (default: 200)
        seed: Random seed for reproducibility (default: 42)
        job_name: Optional name for tracking the job

    Returns:
        Dictionary with job_id for tracking the MSA-enhanced prediction
    """
    script_path = str(SCRIPTS_DIR / "predict_with_msas.py")

    args = {
        "input": input_file,
        "output": output_dir,
        "device": device,
        "num-trunk-recycles": num_trunk_recycles,
        "num-diffn-timesteps": num_diffn_timesteps,
        "seed": seed
    }

    if msa_directory:
        args["msa-dir"] = msa_directory
    if use_msa_server:
        args["use-msa-server"] = ""  # Flag argument

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "msa_prediction"
    )

@mcp.tool()
def submit_batch_prediction(
    input_dir: str,
    output_dir: Optional[str] = None,
    file_pattern: str = "*.fasta",
    parallel: bool = False,
    max_workers: int = 2,
    device: str = "cuda:0",
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch processing of multiple FASTA files for background processing.

    Processes multiple protein sequences in a single batch job.
    Suitable for large-scale analysis and parallel processing.
    Typical runtime: 10+ minutes to hours depending on batch size.

    Args:
        input_dir: Directory containing FASTA files to process
        output_dir: Directory to save all outputs (optional)
        file_pattern: File pattern to match (default: *.fasta)
        parallel: Enable parallel processing of files (default: False)
        max_workers: Number of parallel workers when parallel=True (default: 2)
        device: Compute device (default: cuda:0)
        num_trunk_recycles: Model recycles for accuracy (default: 3)
        num_diffn_timesteps: Diffusion timesteps (default: 200)
        seed: Random seed for reproducibility (default: 42)
        job_name: Optional name for tracking the batch job

    Returns:
        Dictionary with job_id for tracking the batch processing
    """
    script_path = str(SCRIPTS_DIR / "predict_batch_structures.py")

    args = {
        "input-dir": input_dir,
        "output-dir": output_dir,
        "file-pattern": file_pattern,
        "device": device,
        "num-trunk-recycles": num_trunk_recycles,
        "num-diffn-timesteps": num_diffn_timesteps,
        "seed": seed,
        "max-workers": max_workers
    }

    if parallel:
        args["parallel"] = ""  # Flag argument

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_{file_pattern}"
    )

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()