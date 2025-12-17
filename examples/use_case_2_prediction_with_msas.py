#!/usr/bin/env python3
"""
Use Case 2: Structure Prediction with MSAs using Chai-1

This script demonstrates structure prediction using MSAs (Multiple Sequence Alignments) for improved
accuracy. MSAs provide evolutionary information that significantly improves prediction quality.

Usage:
    python examples/use_case_2_prediction_with_msas.py [--input FASTA] [--msa-dir DIR] [--output DIR] [--use-msa-server]
"""

import argparse
import logging
import shutil
import sys
import tempfile
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


def predict_structure_with_msas(
    fasta_path: Path,
    output_dir: Path,
    msa_directory: Path = None,
    use_msa_server: bool = False,
    msa_server_url: str = None,
    use_templates_server: bool = False,
    device: str = "cuda:0"
):
    """
    Predict structure using Chai-1 with MSAs.

    Args:
        fasta_path: Path to input FASTA file
        output_dir: Directory to save output files
        msa_directory: Directory containing .aligned.pqt MSA files
        use_msa_server: Whether to use online MSA server (MMseqs2)
        msa_server_url: Custom MSA server URL (optional)
        use_templates_server: Whether to use template server
        device: Device to use for computation

    Returns:
        Prediction results with CIF paths and scores
    """
    logging.info(f"Starting structure prediction with MSAs for {fasta_path}")
    logging.info(f"Output will be saved to {output_dir}")
    logging.info(f"MSA directory: {msa_directory}")
    logging.info(f"Use MSA server: {use_msa_server}")
    logging.info(f"Use templates server: {use_templates_server}")
    logging.info(f"Using device: {device}")

    # Inference expects an empty directory; enforce this
    if output_dir.exists():
        logging.warning(f"Removing old output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Prepare MSA parameters
        msa_kwargs = {}

        if msa_directory and msa_directory.exists():
            msa_kwargs['msa_directory'] = msa_directory
            msa_kwargs['use_msa_server'] = False
            logging.info(f"Using local MSA files from {msa_directory}")
        elif use_msa_server:
            msa_kwargs['use_msa_server'] = True
            msa_kwargs['use_templates_server'] = use_templates_server
            if msa_server_url:
                msa_kwargs['msa_server_url'] = msa_server_url
            logging.info("Using online MSA server (this may take several minutes)")
        else:
            logging.warning("No MSA source specified, running without MSAs")

        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir,
            # Default settings for good quality prediction
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device=device,
            use_esm_embeddings=True,
            **msa_kwargs
        )

        logging.info("Structure prediction with MSAs completed successfully!")

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
        description="Structure prediction with MSAs using Chai-1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use sample data with online MSA server
    python examples/use_case_2_prediction_with_msas.py --use-msa-server

    # Use sample data with online MSA server and templates
    python examples/use_case_2_prediction_with_msas.py --use-msa-server --use-templates-server

    # Use local MSA files
    python examples/use_case_2_prediction_with_msas.py --msa-dir examples/data/msas/

    # Custom FASTA with MSA server
    python examples/use_case_2_prediction_with_msas.py --input my_protein.fasta --use-msa-server

    # Custom MSA server
    python examples/use_case_2_prediction_with_msas.py --use-msa-server --msa-server-url "https://api.colabfold.com"
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input FASTA file (if not provided, uses sample data)"
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        help="Directory containing .aligned.pqt MSA files"
    )
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help="Use online MSA server (MMseqs2) - requires internet"
    )
    parser.add_argument(
        "--msa-server-url",
        help="Custom MSA server URL (default uses ColabFold public server)"
    )
    parser.add_argument(
        "--use-templates-server",
        action="store_true",
        help="Use template server along with MSA server"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/msa_prediction"),
        help="Output directory (default: outputs/msa_prediction)"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use: cuda:0, cuda:1, or cpu (default: cuda:0)"
    )

    args = parser.parse_args()

    if not args.msa_dir and not args.use_msa_server:
        logging.error("Must specify either --msa-dir or --use-msa-server")
        return 1

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
            fasta_path = Path("examples/data/sample_msa.fasta")
            fasta_path.parent.mkdir(parents=True, exist_ok=True)
            fasta_path.write_text(create_sample_fasta())
            logging.info(f"Created sample FASTA: {fasta_path}")

        # Validate MSA directory if specified
        if args.msa_dir and not args.msa_dir.exists():
            logging.error(f"MSA directory not found: {args.msa_dir}")
            return 1

        # Run prediction
        results = predict_structure_with_msas(
            fasta_path=fasta_path,
            output_dir=args.output,
            msa_directory=args.msa_dir,
            use_msa_server=args.use_msa_server,
            msa_server_url=args.msa_server_url,
            use_templates_server=args.use_templates_server,
            device=args.device
        )

        logging.info("\n" + "="*60)
        logging.info("PREDICTION WITH MSAs COMPLETE")
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