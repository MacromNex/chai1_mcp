# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chai-1 MCP server — provides protein structure prediction via the Model Context Protocol (MCP) using Chai Discovery's Chai-1 model. Supports proteins, DNA, RNA, and ligands with both synchronous and async (job-based) APIs.

## Architecture

**MCP Server** (`src/server.py`): FastMCP-based server exposing 11 tools in three categories:
- **Sync tools** (< 3 min): `validate_fasta_file`, `predict_small_peptide`, `analyze_sequence_composition`
- **Submit tools** (background jobs): `submit_basic_prediction`, `submit_msa_prediction`, `submit_batch_prediction`
- **Job management**: `get_job_status`, `get_job_result`, `get_job_log`, `cancel_job`, `list_jobs`

**Job Manager** (`src/jobs/manager.py`): Handles async execution via threading + subprocess. Jobs are stored in `jobs/<job_id>/` with metadata.json, output.json, and job.log files.

**Scripts** (`scripts/`): Standalone prediction scripts that can be run directly or invoked by the job manager. Each script uses lazy loading for `chai_lab` imports and follows the pattern: DEFAULT_CONFIG → config file merge → CLI override.

**Shared Library** (`scripts/lib/`): `io.py` for FASTA/JSON operations, `utils.py` for device validation, config merging, score extraction, and timing utilities.

## Key Commands

```bash
# Setup (conda)
bash quick_setup.sh

# Run MCP server
env/bin/python src/server.py

# Run prediction scripts directly
env/bin/python scripts/predict_basic_structure.py --input examples/data/simple_test.fasta --output results/
env/bin/python scripts/predict_with_msas.py --input examples/data/sample.fasta --msa-dir examples/data/ --output results/
env/bin/python scripts/predict_batch_structures.py --input-dir examples/data/batch_test --output-dir results/

# Validate environment
env/bin/python scripts/validate_use_cases.py

# Docker
docker build -t chai1_mcp .
docker run --gpus all chai1_mcp

# Register in Claude Code
claude mcp add chai1 -- docker run --gpus all -i --rm ghcr.io/macromnex/chai1_mcp:latest
```

## Key Dependencies

- `chai_lab==0.6.1` — core prediction model
- `fastmcp` — MCP server framework (v3.0)
- `pytorch 2.4.0` + CUDA 11.8 — computation backend

## Configuration

Configs in `configs/` are JSON files merged with priority: defaults → config file → CLI args. Key model params: `num_trunk_recycles` (default 3), `num_diffn_timesteps` (default 200), `seed` (default 42), `device` (default cuda:0).

## Important Patterns

- Heavy dependencies (`chai_lab`) are always lazy-loaded to keep server startup fast
- Submit tools delegate to scripts via subprocess (not direct function calls) so jobs can be cancelled
- The `.gitignore` uses `/jobs/` with `!src/jobs/` to track the job manager module while ignoring runtime job data
- FASTA validation uses `scripts/lib/io.py` — sequences are checked for valid amino acid characters (ACDEFGHIKLMNPQRSTVWY)
