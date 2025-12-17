"""Shared utility functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time


def get_chai_inference():
    """Lazy load chai_lab to minimize startup time."""
    try:
        from chai_lab.chai1 import run_inference
        return run_inference
    except ImportError as e:
        raise ImportError(
            "chai_lab not found. Please install it with: pip install chai_lab==0.6.1"
        ) from e


def extract_scores_from_candidates(candidates) -> List[float]:
    """Extract aggregate scores from candidates object."""
    return [rd.aggregate_score.item() for rd in candidates.ranking_data]


def extract_cif_paths_from_candidates(candidates) -> List[str]:
    """Extract CIF file paths from candidates object."""
    return [str(path) for path in candidates.cif_paths]


def validate_device(device: str) -> str:
    """Validate and normalize device string."""
    device = device.lower().strip()

    if device == "cpu":
        return "cpu"
    elif device.startswith("cuda"):
        # Validate CUDA format
        if ":" in device:
            try:
                device_id = int(device.split(":")[1])
                return f"cuda:{device_id}"
            except (ValueError, IndexError):
                raise ValueError(f"Invalid CUDA device format: {device}")
        else:
            return "cuda:0"  # Default to first GPU
    else:
        raise ValueError(f"Invalid device: {device}. Use 'cpu' or 'cuda:N'")


def merge_configs(default: Dict[str, Any], user: Optional[Dict[str, Any]] = None, **overrides) -> Dict[str, Any]:
    """Merge configuration dictionaries with precedence: overrides > user > default."""
    result = default.copy()

    if user:
        result.update(user)

    if overrides:
        result.update(overrides)

    return result


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h{minutes}m{secs:.1f}s"


def calculate_success_rate(results: List[tuple]) -> Dict[str, Any]:
    """Calculate success rate from batch results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "successful": 0, "failed": 0, "rate": 0.0}

    # Assuming results format: (path, success, error, time, ...)
    successful = sum(1 for result in results if len(result) >= 2 and result[1])
    failed = total - successful

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "rate": (successful / total) * 100 if total > 0 else 0.0
    }


def create_sample_fasta_content() -> str:
    """Create standard sample FASTA content for testing."""
    return """
>protein|name=example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=example-peptide
GAAL
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()


def create_simple_test_fasta() -> str:
    """Create minimal test FASTA for quick testing."""
    return """
>protein|name=test-peptide
GAAL
""".strip()


class Timer:
    """Simple timer context manager."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def __str__(self):
        return format_time(self.elapsed)


def validate_output_results(output_dir: Path) -> Dict[str, Any]:
    """Validate prediction output files and return summary."""
    if not output_dir.exists():
        return {"valid": False, "error": "Output directory does not exist"}

    # Check for CIF files
    cif_files = list(output_dir.glob("*.cif"))
    npz_files = list(output_dir.glob("*.npz"))

    if not cif_files:
        return {"valid": False, "error": "No CIF files found in output"}

    return {
        "valid": True,
        "cif_files": len(cif_files),
        "npz_files": len(npz_files),
        "output_dir": str(output_dir)
    }