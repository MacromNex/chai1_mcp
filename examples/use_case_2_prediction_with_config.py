#!/usr/bin/env python3
"""
Use Case 2: Structure Prediction with Configuration File

This script performs structure prediction using Chai-1 with advanced configuration
options specified in a JSON config file. Supports MSA usage, templates, and
custom prediction parameters.

Usage:
    python use_case_2_prediction_with_config.py --config examples/data/config.json --output outputs/config_prediction
    python use_case_2_prediction_with_config.py --create-config --output outputs/example_config_prediction
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from chai_lab.chai1 import run_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_example_config(config_path):
    """Create an example configuration file."""
    config = {
        "fasta_content": """
>protein|name=antibody-heavy-chain
EVQLVESGGGLIQPGGSLRLSCAASEFIVSRNYMSWVRQAPGTGLEWVSVIYPGGSTFYADSVKGRFTISRDNSKNTLYLQMDSLRVEDTAVYYCARDYGDFYFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDK
>protein|name=antibody-light-chain
EIVMTQSPVSLSVSPGERATLSCRASQGVASNLAWYQQKAGQAPRLLIYGASTRATGIPARFSGSGSGTEFTLTISTLQSEDSAVYYCQQYNDRPRTFGQGTKLEIKRT
""".strip(),
        "prediction_params": {
            "num_trunk_recycles": 3,
            "num_diffn_timesteps": 200,
            "seed": 1234,
            "device": "cuda:0",
            "use_esm_embeddings": True
        },
        "msa_params": {
            "use_msa_server": True,
            "msa_directory": None
        },
        "template_params": {
            "use_templates_server": True,
            "template_directory": None
        },
        "output_params": {
            "num_samples": 5,
            "save_intermediate": False
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Created example configuration file: {config_path}")


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from: {config_path}")
    return config


def predict_with_config(config, output_dir):
    """
    Predict protein structure using configuration parameters.

    Args:
        config (dict): Configuration dictionary
        output_dir (Path): Output directory for results

    Returns:
        Tuple of (cif_paths, aggregate_scores)
    """
    logger.info("Starting structure prediction with configuration...")

    # Create FASTA file from config
    fasta_file = output_dir / "input.fasta"
    fasta_file.parent.mkdir(parents=True, exist_ok=True)
    fasta_file.write_text(config["fasta_content"])

    # Ensure output directory is clean
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.warning(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Re-create FASTA file after cleaning
    fasta_file.write_text(config["fasta_content"])

    # Prepare inference parameters
    inference_params = {
        "fasta_file": fasta_file,
        "output_dir": output_dir,
        **config.get("prediction_params", {}),
        **config.get("msa_params", {}),
        **config.get("template_params", {})
    }

    # Log configuration
    logger.info("Configuration parameters:")
    for key, value in inference_params.items():
        if key not in ["fasta_file", "output_dir"]:
            logger.info(f"  {key}: {value}")

    # Run inference
    candidates = run_inference(**inference_params)

    cif_paths = candidates.cif_paths
    agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

    logger.info(f"Prediction completed. Generated {len(cif_paths)} structures.")

    # Load and display detailed scores
    if len(cif_paths) > 0:
        for i, (cif_path, score) in enumerate(zip(cif_paths, agg_scores)):
            scores_file = output_dir / f"scores.model_idx_{i}.npz"
            if scores_file.exists():
                scores = np.load(scores_file)
                logger.info(f"Model {i+1} ({cif_path.name}):")
                logger.info(f"  Aggregate score: {score:.3f}")
                if 'pae' in scores:
                    logger.info(f"  PAE: {scores['pae'].mean():.3f}")
                if 'plddt' in scores:
                    logger.info(f"  pLDDT: {scores['plddt'].mean():.3f}")
                if 'iptm' in scores:
                    logger.info(f"  ipTM: {scores['iptm']:.3f}")

    return cif_paths, agg_scores


def main():
    parser = argparse.ArgumentParser(
        description="Structure prediction with Chai-1 using configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use existing config file
    python use_case_2_prediction_with_config.py --config examples/data/config.json --output outputs/config_prediction

    # Create and use example config
    python use_case_2_prediction_with_config.py --create-config --output outputs/example_config

    # Use config with custom output directory
    python use_case_2_prediction_with_config.py --config my_config.json --output my_custom_output

Configuration file format:
    {
        "fasta_content": "FASTA format sequences here",
        "prediction_params": {
            "num_trunk_recycles": 3,
            "num_diffn_timesteps": 200,
            "seed": 1234,
            "device": "cuda:0",
            "use_esm_embeddings": true
        },
        "msa_params": {
            "use_msa_server": true,
            "msa_directory": null
        },
        "template_params": {
            "use_templates_server": true,
            "template_directory": null
        },
        "output_params": {
            "num_samples": 5,
            "save_intermediate": false
        }
    }
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration JSON file"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for results"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create example configuration file instead of using existing config"
    )

    args = parser.parse_args()

    # Handle config creation
    if args.create_config:
        config_file = args.output.parent / "example_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        create_example_config(config_file)
        args.config = config_file

    # Validate input
    if not args.config:
        parser.error("Either --config must be specified or --create-config must be used")

    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    try:
        # Load configuration
        config = load_config(args.config)

        # Run prediction
        cif_paths, scores = predict_with_config(config, args.output)

        logger.info(f"Structure prediction completed successfully!")
        logger.info(f"Output files saved to: {args.output}")
        logger.info(f"Generated structures:")
        for i, (cif_path, score) in enumerate(zip(cif_paths, scores)):
            logger.info(f"  {i+1}. {cif_path.name} (score: {score:.3f})")

        # Save summary
        summary = {
            "config_file": str(args.config),
            "output_directory": str(args.output),
            "num_structures": len(cif_paths),
            "structure_files": [str(p) for p in cif_paths],
            "aggregate_scores": scores
        }

        summary_file = args.output / "prediction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {summary_file}")
        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())