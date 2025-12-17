#!/usr/bin/env python3
"""
Use Case 1: Basic Structure Prediction from FASTA sequence with Chai-1

This script demonstrates basic structure prediction using Chai-1 from a FASTA sequence containing
proteins and ligands. This is the fundamental use case for most users.

Usage:
    python examples/use_case_1_basic_structure_prediction.py [--input FASTA] [--output DIR] [--device DEVICE]
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from chai_lab.chai1 import run_inference
except ImportError:
    logging.error("chai_lab not found. Please install it with: pip install chai_lab==0.6.1")
    sys.exit(1)


def create_sample_fasta():
    """Create a sample FASTA file for testing."""
    sample_content = """
>protein|name=example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=example-peptide
GAAL
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()
    return sample_content


def predict_structure(fasta_path: Path, output_dir: Path, device: str = "cuda:0"):
    """
    Predict structure using Chai-1.

    Args:
        fasta_path: Path to input FASTA file
        output_dir: Directory to save output files
        device: Device to use for computation (cuda:0, cpu)

    Returns:
        Prediction results with CIF paths and scores
    """
    logging.info(f"Starting structure prediction for {fasta_path}")
    logging.info(f"Output will be saved to {output_dir}")
    logging.info(f"Using device: {device}")

    # Inference expects an empty directory; enforce this
    if output_dir.exists():
        logging.warning(f"Removing old output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir,
            # Default settings for good quality prediction
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device=device,
            use_esm_embeddings=True,
        )

        logging.info("Structure prediction completed successfully!")

        # Extract results
        cif_paths = candidates.cif_paths
        agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

        logging.info(f"Generated {len(cif_paths)} structure predictions")
        for i, (path, score) in enumerate(zip(cif_paths, agg_scores)):
            logging.info(f"Prediction {i+1}: {path.name} (score: {score:.4f})")

        # Load detailed scores for the best prediction (model_idx_0)
        try:
            scores_path = output_dir / "scores.model_idx_0.npz"
            if scores_path.exists():
                scores = np.load(scores_path)
                logging.info("Detailed scores loaded for best prediction:")
                for key in scores.files:
                    if hasattr(scores[key], 'shape'):
                        logging.info(f"  {key}: shape {scores[key].shape}")
                    else:
                        logging.info(f"  {key}: {scores[key]}")
        except Exception as e:
            logging.warning(f"Could not load detailed scores: {e}")

        return candidates

    except Exception as e:
        logging.error(f"Structure prediction failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Basic structure prediction using Chai-1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use sample data
    python examples/use_case_1_basic_structure_prediction.py

    # Predict structure from custom FASTA
    python examples/use_case_1_basic_structure_prediction.py --input my_protein.fasta --output results/

    # Use CPU instead of GPU
    python examples/use_case_1_basic_structure_prediction.py --device cpu
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input FASTA file (if not provided, uses sample data)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/basic_prediction"),
        help="Output directory (default: outputs/basic_prediction)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use: cuda:0, cuda:1, or cpu (default: cuda:0)"
    )

    args = parser.parse_args()

    try:
        # Handle input FASTA
        if args.input:
            if not args.input.exists():
                logging.error(f"Input file not found: {args.input}")
                return 1
            fasta_path = args.input
            logging.info(f"Using input FASTA: {fasta_path}")
        else:
            # Create sample data
            fasta_path = Path("examples/data/sample_basic.fasta")
            fasta_path.parent.mkdir(parents=True, exist_ok=True)
            fasta_path.write_text(create_sample_fasta())
            logging.info(f"Created sample FASTA: {fasta_path}")

        # Run prediction
        results = predict_structure(fasta_path, args.output, args.device)

        logging.info("\n" + "="*60)
        logging.info("PREDICTION COMPLETE")
        logging.info("="*60)
        logging.info(f"Results saved to: {args.output}")
        logging.info(f"Best prediction: {results.cif_paths[0]}")
        logging.info(f"Total predictions: {len(results.cif_paths)}")

        return 0

    except KeyboardInterrupt:
        logging.info("Prediction interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())