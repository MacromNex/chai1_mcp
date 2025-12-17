#!/usr/bin/env python3
"""
Use Case 1: Basic Structure Prediction Given a Sequence with Chai-1

This script performs basic structure prediction using Chai-1 with sequences provided
in FASTA format. It supports proteins, ligands, DNA, and RNA.

Usage:
    python use_case_1_basic_prediction.py --input examples/data/sample.fasta --output outputs/basic_prediction
    python use_case_1_basic_prediction.py --input examples/data/protein_ligand.fasta --output outputs/protein_ligand
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from chai_lab.chai1 import run_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_example_fasta(output_path):
    """Create an example FASTA file with protein and ligand sequences."""
    example_fasta = """
>protein|name=example-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=peptide
GAAL
>ligand|name=fatty-acid
CCCCCCCCCCCCCC(=O)O
""".strip()

    output_path.write_text(example_fasta)
    logger.info(f"Created example FASTA file: {output_path}")


def predict_structure(fasta_file, output_dir, device="cuda:0", num_samples=5, seed=42):
    """
    Predict protein structure using Chai-1.

    Args:
        fasta_file (Path): Path to input FASTA file
        output_dir (Path): Output directory for results
        device (str): Device to use (cuda:0, cpu)
        num_samples (int): Number of prediction samples to generate
        seed (int): Random seed for reproducibility

    Returns:
        Tuple of (cif_paths, aggregate_scores)
    """
    logger.info(f"Starting structure prediction with Chai-1...")
    logger.info(f"Input: {fasta_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Samples: {num_samples}")

    # Ensure output directory is clean
    if output_dir.exists():
        logger.warning(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    candidates = run_inference(
        fasta_file=fasta_file,
        output_dir=output_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        seed=seed,
        device=device,
        use_esm_embeddings=True,
    )

    cif_paths = candidates.cif_paths
    agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

    logger.info(f"Prediction completed. Generated {len(cif_paths)} structures.")

    # Load and display scores for the best model
    if len(cif_paths) > 0:
        best_model_idx = 0  # Models are already ranked by score
        scores_file = output_dir / f"scores.model_idx_{best_model_idx}.npz"

        if scores_file.exists():
            scores = np.load(scores_file)
            logger.info(f"Best model scores:")
            logger.info(f"  Aggregate score: {agg_scores[best_model_idx]:.3f}")
            if 'pae' in scores:
                logger.info(f"  PAE: {scores['pae'].mean():.3f}")
            if 'plddt' in scores:
                logger.info(f"  pLDDT: {scores['plddt'].mean():.3f}")

    return cif_paths, agg_scores


def main():
    parser = argparse.ArgumentParser(
        description="Basic structure prediction with Chai-1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use example data
    python use_case_1_basic_prediction.py --input examples/data/sample.fasta --output outputs/basic_prediction

    # Create and use example FASTA
    python use_case_1_basic_prediction.py --create-example --output outputs/example_prediction

    # Use custom parameters
    python use_case_1_basic_prediction.py --input my_protein.fasta --output my_output --device cpu --seed 123
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input FASTA file with sequences"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (default: cuda:0, use 'cpu' if no GPU)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of prediction samples (default: 5)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create example FASTA file instead of using input file"
    )

    args = parser.parse_args()

    # Handle example creation
    if args.create_example:
        fasta_file = args.output.parent / "example.fasta"
        fasta_file.parent.mkdir(parents=True, exist_ok=True)
        create_example_fasta(fasta_file)
        args.input = fasta_file

    # Validate input
    if not args.input:
        parser.error("Either --input must be specified or --create-example must be used")

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    try:
        # Run prediction
        cif_paths, scores = predict_structure(
            fasta_file=args.input,
            output_dir=args.output,
            device=args.device,
            num_samples=args.samples,
            seed=args.seed
        )

        logger.info(f"Structure prediction completed successfully!")
        logger.info(f"Output files saved to: {args.output}")
        logger.info(f"Generated structures:")
        for i, (cif_path, score) in enumerate(zip(cif_paths, scores)):
            logger.info(f"  {i+1}. {cif_path.name} (score: {score:.3f})")

        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())