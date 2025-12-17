#!/usr/bin/env python3
"""
Script: predict_basic_structure.py
Description: Basic structure prediction from FASTA sequence using Chai-1

Original Use Case: examples/use_case_1_basic_structure_prediction.py
Dependencies Removed: Redundant logging setup, sample data generation inlined

Usage:
    python scripts/predict_basic_structure.py --input <input_file> --output <output_file>

Example:
    python scripts/predict_basic_structure.py --input examples/data/sample.fasta --output results/basic_pred
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import shutil
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any

import numpy as np

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "num_trunk_recycles": 3,
    "num_diffn_timesteps": 200,
    "seed": 42,
    "use_esm_embeddings": True,
    "device": "cuda:0"
}

# ==============================================================================
# Lazy Import for Heavy Dependencies
# ==============================================================================
def get_chai_inference():
    """Lazy load chai_lab to minimize startup time."""
    try:
        from chai_lab.chai1 import run_inference
        return run_inference
    except ImportError as e:
        raise ImportError(
            "chai_lab not found. Please install it with: pip install chai_lab==0.6.1"
        ) from e

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def create_sample_fasta_content() -> str:
    """Create sample FASTA content for testing."""
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

def load_input(file_path: Path) -> Path:
    """Load and validate input FASTA file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return file_path

def save_output_metadata(data: Dict[str, Any], output_dir: Path) -> None:
    """Save prediction metadata to output directory."""
    metadata_file = output_dir / "prediction_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Chai-1 Basic Structure Prediction Results\n")
        f.write("=" * 45 + "\n")
        f.write(f"Input file: {data.get('input_file', 'N/A')}\n")
        f.write(f"Output directory: {data.get('output_dir', 'N/A')}\n")
        f.write(f"Device used: {data.get('device', 'N/A')}\n")
        f.write(f"Number of predictions: {data.get('num_predictions', 'N/A')}\n")
        f.write(f"Best prediction score: {data.get('best_score', 'N/A')}\n")
        f.write(f"Best prediction file: {data.get('best_prediction', 'N/A')}\n")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_basic_structure(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for basic structure prediction using Chai-1.

    Args:
        input_file: Path to input FASTA file
        output_dir: Path to save output (optional, auto-generated if not provided)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - cif_paths: List of CIF file paths
            - scores: List of aggregate scores
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_predict_basic_structure("input.fasta", "output_dir/")
        >>> print(result['output_dir'])
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load input
    input_path = load_input(input_file)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"results/basic_prediction_{input_file.stem}")
    else:
        output_dir = Path(output_dir)

    # Ensure clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy load chai inference
    run_inference = get_chai_inference()

    try:
        # Core processing (extracted and simplified from use case)
        candidates = run_inference(
            fasta_file=input_path,
            output_dir=output_dir,
            num_trunk_recycles=config["num_trunk_recycles"],
            num_diffn_timesteps=config["num_diffn_timesteps"],
            seed=config["seed"],
            device=config["device"],
            use_esm_embeddings=config["use_esm_embeddings"],
        )

        # Extract results
        cif_paths = candidates.cif_paths
        agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

        # Save metadata
        metadata = {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "device": config["device"],
            "num_predictions": len(cif_paths),
            "best_score": agg_scores[0] if agg_scores else None,
            "best_prediction": str(cif_paths[0]) if cif_paths else None,
            "config": config
        }

        save_output_metadata(metadata, output_dir)

        return {
            "cif_paths": [str(path) for path in cif_paths],
            "scores": agg_scores,
            "output_dir": str(output_dir),
            "metadata": metadata
        }

    except Exception as e:
        raise RuntimeError(f"Structure prediction failed: {e}") from e

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file path')
    parser.add_argument('--output', '-o', help='Output directory path')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--device', help='Device to use (cuda:0, cpu)', default="cuda:0")
    parser.add_argument('--num-trunk-recycles', type=int, help='Number of trunk recycles')
    parser.add_argument('--num-diffn-timesteps', type=int, help='Number of diffusion timesteps')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI args
    cli_overrides = {}
    if args.device:
        cli_overrides['device'] = args.device
    if args.num_trunk_recycles is not None:
        cli_overrides['num_trunk_recycles'] = args.num_trunk_recycles
    if args.num_diffn_timesteps is not None:
        cli_overrides['num_diffn_timesteps'] = args.num_diffn_timesteps
    if args.seed is not None:
        cli_overrides['seed'] = args.seed

    # Run
    try:
        result = run_predict_basic_structure(
            input_file=args.input,
            output_dir=args.output,
            config=config,
            **cli_overrides
        )

        print(f"✅ Success: {len(result['cif_paths'])} predictions generated")
        print(f"   Output directory: {result['output_dir']}")
        print(f"   Best prediction: {result['metadata']['best_prediction']}")
        print(f"   Best score: {result['metadata']['best_score']:.4f}")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())