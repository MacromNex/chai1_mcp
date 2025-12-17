#!/usr/bin/env python3
"""
Script: predict_with_msas.py
Description: Structure prediction with MSAs (Multiple Sequence Alignments) using Chai-1

Original Use Case: examples/use_case_2_prediction_with_msas.py
Dependencies Removed: Redundant logging setup, sample data generation inlined

Usage:
    python scripts/predict_with_msas.py --input <input_file> --output <output_file> [--msa-dir DIR | --use-msa-server]

Example:
    python scripts/predict_with_msas.py --input examples/data/sample.fasta --output results/msa_pred --msa-dir examples/data/
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
    "device": "cuda:0",
    "use_msa_server": False,
    "use_templates_server": False,
    "msa_server_url": None
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

def validate_msa_directory(msa_dir: Path) -> bool:
    """Check if MSA directory contains valid MSA files."""
    if not msa_dir.exists():
        return False
    msa_files = list(msa_dir.glob("*.aligned.pqt"))
    return len(msa_files) > 0

def save_output_metadata(data: Dict[str, Any], output_dir: Path) -> None:
    """Save prediction metadata to output directory."""
    metadata_file = output_dir / "prediction_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Chai-1 MSA-Enhanced Structure Prediction Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input file: {data.get('input_file', 'N/A')}\n")
        f.write(f"Output directory: {data.get('output_dir', 'N/A')}\n")
        f.write(f"Device used: {data.get('device', 'N/A')}\n")
        f.write(f"MSA source: {data.get('msa_source', 'N/A')}\n")
        f.write(f"Number of predictions: {data.get('num_predictions', 'N/A')}\n")
        f.write(f"Best prediction score: {data.get('best_score', 'N/A')}\n")
        f.write(f"Best prediction file: {data.get('best_prediction', 'N/A')}\n")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_with_msas(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    msa_directory: Optional[Union[str, Path]] = None,
    use_msa_server: bool = False,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for MSA-enhanced structure prediction using Chai-1.

    Args:
        input_file: Path to input FASTA file
        output_dir: Path to save output (optional, auto-generated if not provided)
        msa_directory: Directory containing .aligned.pqt MSA files
        use_msa_server: Whether to use online MSA server
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - cif_paths: List of CIF file paths
            - scores: List of aggregate scores
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_predict_with_msas("input.fasta", msa_directory="msas/")
        >>> print(result['output_dir'])
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load input
    input_path = load_input(input_file)

    # Setup output directory
    if output_dir is None:
        msa_suffix = "server" if use_msa_server else "local"
        output_dir = Path(f"results/msa_prediction_{input_file.stem}_{msa_suffix}")
    else:
        output_dir = Path(output_dir)

    # Ensure clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare MSA parameters
    msa_kwargs = {}
    msa_source = "none"

    if msa_directory:
        msa_dir = Path(msa_directory)
        if not validate_msa_directory(msa_dir):
            raise ValueError(f"MSA directory does not exist or contains no .aligned.pqt files: {msa_dir}")
        msa_kwargs['msa_directory'] = msa_dir
        msa_kwargs['use_msa_server'] = False
        msa_source = f"local:{msa_dir}"
    elif use_msa_server:
        msa_kwargs['use_msa_server'] = True
        msa_kwargs['use_templates_server'] = config.get('use_templates_server', False)
        if config.get('msa_server_url'):
            msa_kwargs['msa_server_url'] = config['msa_server_url']
        msa_source = "online_server"
    else:
        raise ValueError("Must specify either msa_directory or use_msa_server=True")

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
            **msa_kwargs
        )

        # Extract results
        cif_paths = candidates.cif_paths
        agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

        # Save metadata
        metadata = {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "device": config["device"],
            "msa_source": msa_source,
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
        raise RuntimeError(f"MSA-enhanced structure prediction failed: {e}") from e

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
    parser.add_argument('--msa-dir', help='Directory containing .aligned.pqt MSA files')
    parser.add_argument('--use-msa-server', action='store_true', help='Use online MSA server')
    parser.add_argument('--use-templates-server', action='store_true', help='Use templates server with MSA server')
    parser.add_argument('--msa-server-url', help='Custom MSA server URL')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--device', help='Device to use (cuda:0, cpu)', default="cuda:0")
    parser.add_argument('--num-trunk-recycles', type=int, help='Number of trunk recycles')
    parser.add_argument('--num-diffn-timesteps', type=int, help='Number of diffusion timesteps')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    # Validate MSA options
    if not args.msa_dir and not args.use_msa_server:
        print("❌ Error: Must specify either --msa-dir or --use-msa-server")
        return 1

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
    if args.use_templates_server:
        cli_overrides['use_templates_server'] = True
    if args.msa_server_url:
        cli_overrides['msa_server_url'] = args.msa_server_url
    if args.num_trunk_recycles is not None:
        cli_overrides['num_trunk_recycles'] = args.num_trunk_recycles
    if args.num_diffn_timesteps is not None:
        cli_overrides['num_diffn_timesteps'] = args.num_diffn_timesteps
    if args.seed is not None:
        cli_overrides['seed'] = args.seed

    # Run
    try:
        result = run_predict_with_msas(
            input_file=args.input,
            output_dir=args.output,
            msa_directory=args.msa_dir,
            use_msa_server=args.use_msa_server,
            config=config,
            **cli_overrides
        )

        print(f"✅ Success: {len(result['cif_paths'])} predictions generated")
        print(f"   Output directory: {result['output_dir']}")
        print(f"   MSA source: {result['metadata']['msa_source']}")
        print(f"   Best prediction: {result['metadata']['best_prediction']}")
        print(f"   Best score: {result['metadata']['best_score']:.4f}")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())