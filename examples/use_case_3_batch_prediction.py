#!/usr/bin/env python3
"""
Use Case 3: Batch Structure Prediction with Chai-1

This script demonstrates batch processing of multiple FASTA files for structure prediction.
It's useful for processing multiple proteins or protein complexes in a single workflow.

Usage:
    python examples/use_case_3_batch_prediction.py [--input-dir DIR] [--pattern PATTERN] [--output-dir DIR] [--parallel]
"""

import argparse
import concurrent.futures
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from chai_lab.chai1 import run_inference
except ImportError:
    logging.error("chai_lab not found. Please install it with: pip install chai_lab==0.6.1")
    sys.exit(1)


def create_sample_fastas(output_dir: Path) -> List[Path]:
    """Create sample FASTA files for batch testing."""
    samples = {
        "protein_complex.fasta": """
>protein|name=protein-A
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=protein-B
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>ligand|name=ligand-A
CCCCCCCCCCCCCC(=O)O
""".strip(),
        "single_protein.fasta": """
>protein|name=single-domain-protein
GSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
""".strip(),
        "peptide_ligand.fasta": """
>protein|name=short-peptide
GAAL
>ligand|name=small-molecule
CC(C)C(=O)O
""".strip()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for filename, content in samples.items():
        file_path = output_dir / filename
        file_path.write_text(content)
        created_files.append(file_path)
        logging.info(f"Created sample file: {file_path}")

    return created_files


def predict_single_structure(
    fasta_path: Path,
    base_output_dir: Path,
    device: str = "cuda:0",
    use_msa_server: bool = False
) -> Tuple[Path, bool, str, float]:
    """
    Predict structure for a single FASTA file.

    Args:
        fasta_path: Path to FASTA file
        base_output_dir: Base output directory
        device: Device to use for computation
        use_msa_server: Whether to use MSA server

    Returns:
        Tuple of (fasta_path, success, error_message, prediction_time)
    """
    start_time = time.time()
    output_subdir = base_output_dir / fasta_path.stem

    try:
        logging.info(f"Processing {fasta_path.name}...")

        # Prepare output directory
        if output_subdir.exists():
            shutil.rmtree(output_subdir)
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Run prediction
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=output_subdir,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device=device,
            use_esm_embeddings=True,
            use_msa_server=use_msa_server,
            use_templates_server=use_msa_server,  # Use templates when using MSA server
        )

        end_time = time.time()
        prediction_time = end_time - start_time

        # Log results
        agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
        logging.info(f"✓ {fasta_path.name}: {len(candidates.cif_paths)} predictions "
                    f"(best score: {agg_scores[0]:.4f}, time: {prediction_time:.1f}s)")

        return fasta_path, True, "", prediction_time

    except Exception as e:
        end_time = time.time()
        prediction_time = end_time - start_time
        error_msg = str(e)
        logging.error(f"✗ {fasta_path.name}: Failed - {error_msg}")
        return fasta_path, False, error_msg, prediction_time


def batch_predict_sequential(
    fasta_files: List[Path],
    output_dir: Path,
    device: str = "cuda:0",
    use_msa_server: bool = False
) -> List[Tuple[Path, bool, str, float]]:
    """Run batch prediction sequentially."""
    logging.info(f"Running batch prediction sequentially on {device}")
    results = []

    for fasta_file in fasta_files:
        result = predict_single_structure(fasta_file, output_dir, device, use_msa_server)
        results.append(result)

    return results


def batch_predict_parallel(
    fasta_files: List[Path],
    output_dir: Path,
    device_prefix: str = "cuda",
    max_workers: int = 2,
    use_msa_server: bool = False
) -> List[Tuple[Path, bool, str, float]]:
    """Run batch prediction in parallel using multiple GPUs."""
    logging.info(f"Running batch prediction in parallel with {max_workers} workers")

    def worker_predict(args):
        fasta_file, worker_id = args
        # Assign device based on worker ID
        if device_prefix == "cuda":
            device = f"cuda:{worker_id % 4}"  # Cycle through 4 GPUs max
        else:
            device = "cpu"

        return predict_single_structure(fasta_file, output_dir, device, use_msa_server)

    # Create worker arguments
    worker_args = [(fasta_file, i) for i, fasta_file in enumerate(fasta_files)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fasta = {
            executor.submit(worker_predict, args): args[0]
            for args in worker_args
        }

        for future in concurrent.futures.as_completed(future_to_fasta):
            result = future.result()
            results.append(result)

    return results


def summarize_results(results: List[Tuple[Path, bool, str, float]], output_dir: Path):
    """Create a summary report of batch prediction results."""
    total_files = len(results)
    successful = sum(1 for _, success, _, _ in results if success)
    failed = total_files - successful
    total_time = sum(time for _, _, _, time in results)

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
        for fasta_path, success, error, time in results:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{fasta_path.name}: {status} ({time:.1f}s)")
            if not success:
                f.write(f" - {error}")
            f.write("\n")

    logging.info(f"\nBatch Summary:")
    logging.info(f"  Total: {total_files}, Success: {successful}, Failed: {failed}")
    logging.info(f"  Success rate: {successful/total_files*100:.1f}%")
    logging.info(f"  Total time: {total_time:.1f}s, Avg: {total_time/total_files:.1f}s/file")
    logging.info(f"  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch structure prediction using Chai-1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process sample data sequentially
    python examples/use_case_3_batch_prediction.py

    # Process all FASTA files in a directory
    python examples/use_case_3_batch_prediction.py --input-dir my_fastas/ --output-dir batch_results/

    # Process with parallel execution (2 GPUs)
    python examples/use_case_3_batch_prediction.py --parallel --max-workers 2

    # Process with MSA server
    python examples/use_case_3_batch_prediction.py --use-msa-server

    # Process specific file pattern
    python examples/use_case_3_batch_prediction.py --input-dir data/ --pattern "*_complex.fasta"
        """
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing FASTA files (if not provided, creates sample data)"
    )
    parser.add_argument(
        "--pattern",
        default="*.fasta",
        help="File pattern to match (default: *.fasta)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch_prediction"),
        help="Output directory for all results (default: outputs/batch_prediction)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run predictions in parallel (requires multiple GPUs)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for sequential mode, or device prefix for parallel (default: cuda:0)"
    )
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help="Use online MSA server for better accuracy (slower)"
    )

    args = parser.parse_args()

    try:
        # Find or create input files
        if args.input_dir:
            if not args.input_dir.exists():
                logging.error(f"Input directory not found: {args.input_dir}")
                return 1
            fasta_files = list(args.input_dir.glob(args.pattern))
            if not fasta_files:
                logging.error(f"No files matching pattern '{args.pattern}' in {args.input_dir}")
                return 1
            logging.info(f"Found {len(fasta_files)} FASTA files in {args.input_dir}")
        else:
            # Create sample data
            sample_dir = Path("examples/data/batch_samples")
            fasta_files = create_sample_fastas(sample_dir)
            logging.info(f"Created {len(fasta_files)} sample FASTA files")

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Run batch prediction
        start_time = time.time()
        if args.parallel:
            device_prefix = args.device.split(':')[0] if ':' in args.device else args.device
            results = batch_predict_parallel(
                fasta_files, args.output_dir, device_prefix, args.max_workers, args.use_msa_server
            )
        else:
            results = batch_predict_sequential(
                fasta_files, args.output_dir, args.device, args.use_msa_server
            )

        end_time = time.time()

        # Summarize results
        summarize_results(results, args.output_dir)

        logging.info("\n" + "="*60)
        logging.info("BATCH PREDICTION COMPLETE")
        logging.info("="*60)
        logging.info(f"Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
        logging.info(f"Total wall time: {end_time - start_time:.1f} seconds")
        logging.info(f"Results saved to: {args.output_dir}")

        return 0

    except KeyboardInterrupt:
        logging.info("Batch prediction interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Batch prediction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())