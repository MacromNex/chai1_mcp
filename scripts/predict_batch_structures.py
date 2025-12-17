#!/usr/bin/env python3
"""
Script: predict_batch_structures.py
Description: Batch structure prediction of multiple FASTA files using Chai-1

Original Use Case: examples/use_case_3_batch_prediction.py
Dependencies Removed: Redundant logging setup, sample data generation inlined

Usage:
    python scripts/predict_batch_structures.py --input-dir <input_dir> --output-dir <output_dir>

Example:
    python scripts/predict_batch_structures.py --input-dir examples/data/batch_test --output-dir results/batch
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import concurrent.futures
import shutil
import sys
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple

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
    "parallel": False,
    "max_workers": 2,
    "use_msa_server": False,
    "file_pattern": "*.fasta"
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
def create_sample_batch_data(output_dir: Path) -> List[Path]:
    """Create sample FASTA files for batch testing."""
    samples = {
        "test1.fasta": """
>protein|name=peptide-test-1
GAAL
""".strip(),
        "test2.fasta": """
>protein|name=peptide-test-2
GAALD
""".strip()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for filename, content in samples.items():
        file_path = output_dir / filename
        file_path.write_text(content)
        created_files.append(file_path)

    return created_files

def find_input_files(input_dir: Path, pattern: str = "*.fasta") -> List[Path]:
    """Find FASTA files in input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = list(input_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files matching pattern '{pattern}' found in {input_dir}")

    return sorted(files)

def predict_single_file(
    fasta_path: Path,
    base_output_dir: Path,
    device: str,
    use_msa_server: bool,
    config: Dict[str, Any]
) -> Tuple[Path, bool, str, float, Optional[float]]:
    """
    Predict structure for a single FASTA file.

    Returns:
        Tuple of (fasta_path, success, error_message, prediction_time, best_score)
    """
    start_time = time.time()
    output_subdir = base_output_dir / fasta_path.stem

    try:
        # Prepare output directory
        if output_subdir.exists():
            shutil.rmtree(output_subdir)
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Lazy load chai inference
        run_inference = get_chai_inference()

        # Run prediction
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=output_subdir,
            num_trunk_recycles=config["num_trunk_recycles"],
            num_diffn_timesteps=config["num_diffn_timesteps"],
            seed=config["seed"],
            device=device,
            use_esm_embeddings=config["use_esm_embeddings"],
            use_msa_server=use_msa_server,
            use_templates_server=use_msa_server,  # Use templates when using MSA server
        )

        end_time = time.time()
        prediction_time = end_time - start_time

        # Extract best score
        agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
        best_score = agg_scores[0] if agg_scores else None

        return fasta_path, True, "", prediction_time, best_score

    except Exception as e:
        end_time = time.time()
        prediction_time = end_time - start_time
        error_msg = str(e)
        return fasta_path, False, error_msg, prediction_time, None

def save_batch_summary(results: List[Tuple], output_dir: Path) -> None:
    """Create a summary report of batch prediction results."""
    total_files = len(results)
    successful = sum(1 for _, success, _, _, _ in results if success)
    failed = total_files - successful
    total_time = sum(time for _, _, _, time, _ in results)

    # Create summary report
    summary_path = output_dir / "batch_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Chai-1 Batch Prediction Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Successful predictions: {successful}\n")
        f.write(f"Failed predictions: {failed}\n")
        f.write(f"Success rate: {successful/total_files*100:.1f}%\n")
        f.write(f"Total processing time: {total_time:.1f} seconds\n")
        f.write(f"Average time per file: {total_time/total_files:.1f} seconds\n\n")

        f.write("Detailed Results:\n")
        f.write("-" * 20 + "\n")
        for fasta_path, success, error, time, best_score in results:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{fasta_path.name}: {status} ({time:.1f}s")
            if success and best_score is not None:
                f.write(f", best score: {best_score:.4f}")
            if not success:
                f.write(f") - {error}")
            else:
                f.write(")")
            f.write("\n")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_batch_structures(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for batch structure prediction using Chai-1.

    Args:
        input_dir: Directory containing FASTA files
        output_dir: Path to save outputs (optional, auto-generated if not provided)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: List of prediction results
            - output_dir: Path to output directory
            - summary_file: Path to summary report
            - metadata: Execution metadata

    Example:
        >>> result = run_predict_batch_structures("batch_input/", "batch_output/")
        >>> print(result['output_dir'])
    """
    # Setup
    input_dir = Path(input_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"results/batch_prediction_{input_dir.name}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input files
    try:
        fasta_files = find_input_files(input_dir, config["file_pattern"])
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Input validation failed: {e}") from e

    # Run predictions
    start_time = time.time()

    if config["parallel"]:
        # Parallel execution
        device_prefix = config["device"].split(':')[0] if ':' in config["device"] else config["device"]

        def worker_predict(args):
            fasta_file, worker_id = args
            # Assign device based on worker ID
            if device_prefix == "cuda":
                device = f"cuda:{worker_id % 4}"  # Cycle through 4 GPUs max
            else:
                device = "cpu"
            return predict_single_file(
                fasta_file, output_dir, device,
                config["use_msa_server"], config
            )

        # Create worker arguments
        worker_args = [(fasta_file, i) for i, fasta_file in enumerate(fasta_files)]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
            future_to_fasta = {
                executor.submit(worker_predict, args): args[0]
                for args in worker_args
            }

            for future in concurrent.futures.as_completed(future_to_fasta):
                result = future.result()
                results.append(result)
    else:
        # Sequential execution
        results = []
        for fasta_file in fasta_files:
            result = predict_single_file(
                fasta_file, output_dir, config["device"],
                config["use_msa_server"], config
            )
            results.append(result)

    end_time = time.time()
    wall_time = end_time - start_time

    # Save summary
    save_batch_summary(results, output_dir)

    # Prepare metadata
    successful = sum(1 for _, success, _, _, _ in results if success)
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_files": len(results),
        "successful": successful,
        "failed": len(results) - successful,
        "wall_time": wall_time,
        "parallel_mode": config["parallel"],
        "config": config
    }

    return {
        "results": results,
        "output_dir": str(output_dir),
        "summary_file": str(output_dir / "batch_summary.txt"),
        "metadata": metadata
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input-dir', required=True, help='Directory containing FASTA files')
    parser.add_argument('--output-dir', help='Output directory for all results')
    parser.add_argument('--pattern', default='*.fasta', help='File pattern to match (default: *.fasta)')
    parser.add_argument('--parallel', action='store_true', help='Run predictions in parallel')
    parser.add_argument('--max-workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--use-msa-server', action='store_true', help='Use online MSA server')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--device', default='cuda:0', help='Device to use (cuda:0, cpu)')
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
    if args.pattern != '*.fasta':
        cli_overrides['file_pattern'] = args.pattern
    if args.parallel:
        cli_overrides['parallel'] = True
    if args.max_workers != 2:
        cli_overrides['max_workers'] = args.max_workers
    if args.use_msa_server:
        cli_overrides['use_msa_server'] = True
    if args.device != 'cuda:0':
        cli_overrides['device'] = args.device
    if args.num_trunk_recycles is not None:
        cli_overrides['num_trunk_recycles'] = args.num_trunk_recycles
    if args.num_diffn_timesteps is not None:
        cli_overrides['num_diffn_timesteps'] = args.num_diffn_timesteps
    if args.seed is not None:
        cli_overrides['seed'] = args.seed

    # Run
    try:
        result = run_predict_batch_structures(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config,
            **cli_overrides
        )

        metadata = result['metadata']
        print(f"✅ Success: Batch prediction completed")
        print(f"   Input directory: {metadata['input_dir']}")
        print(f"   Output directory: {metadata['output_dir']}")
        print(f"   Total files: {metadata['total_files']}")
        print(f"   Successful: {metadata['successful']}")
        print(f"   Failed: {metadata['failed']}")
        print(f"   Success rate: {metadata['successful']/metadata['total_files']*100:.1f}%")
        print(f"   Wall time: {metadata['wall_time']:.1f}s")
        print(f"   Summary: {result['summary_file']}")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())