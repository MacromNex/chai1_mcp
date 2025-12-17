"""Shared I/O functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Any, Dict
import json


def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_fasta_file(file_path: Path) -> bool:
    """Validate FASTA file format."""
    if not file_path.exists():
        return False

    try:
        with open(file_path) as f:
            content = f.read().strip()

        # Check for FASTA format
        lines = content.split('\n')
        has_header = any(line.startswith('>') for line in lines)
        has_sequence = any(line and not line.startswith('>') for line in lines)

        return has_header and has_sequence
    except Exception:
        return False


def read_fasta_content(file_path: Path) -> str:
    """Read FASTA file content."""
    with open(file_path) as f:
        return f.read().strip()


def write_fasta_content(content: str, file_path: Path) -> None:
    """Write FASTA content to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)


def create_output_directory(output_dir: Path, clean: bool = True) -> Path:
    """Create output directory, optionally cleaning it first."""
    if clean and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir