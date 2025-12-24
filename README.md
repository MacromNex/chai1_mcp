# chai-lab MCP

> Protein structure prediction using Chai-1 model via Model Context Protocol (MCP)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

This MCP server provides protein structure prediction capabilities using Chai Discovery's Chai-1 model. It supports proteins, DNA, RNA, and ligands with both quick synchronous operations and long-running background jobs.

### Features
- **Basic Structure Prediction**: Fast prediction from FASTA sequences
- **MSA-Enhanced Prediction**: Improved accuracy using evolutionary information
- **Batch Processing**: High-throughput analysis of multiple structures
- **Job Management**: Background processing with status tracking
- **Validation Tools**: Pre-flight checks and sequence analysis
- **Multi-format Support**: Proteins, DNA, RNA, and ligands (SMILES)

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   └── server.py           # MCP server with 11 tools
├── scripts/
│   ├── predict_basic_structure.py      # Basic structure prediction
│   ├── predict_with_msas.py            # MSA-enhanced prediction
│   ├── predict_batch_structures.py     # Batch processing
│   └── lib/                             # Shared utilities
├── examples/
│   └── data/               # Demo FASTA files and MSA data
├── configs/                # Configuration templates
└── repo/                   # Original chai-lab repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 8-16GB system RAM
- ~10GB disk space for environment

### Create Environment
Please strictly follow the information in `reports/step3_environment.md` to obtain the procedure to setup the environment. An example workflow is shown below.

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/chai1_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install Dependencies
pip install loguru click pandas numpy tqdm
pip install --force-reinstall --no-cache-dir fastmcp
pip install chai_lab==0.6.1
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/predict_basic_structure.py` | Basic structure prediction from FASTA | See below |
| `scripts/predict_with_msas.py` | MSA-enhanced prediction for improved accuracy | See below |
| `scripts/predict_batch_structures.py` | Batch processing of multiple FASTA files | See below |

### Script Examples

#### Basic Structure Prediction

```bash
# Activate environment
mamba activate ./env

# Run basic prediction
python scripts/predict_basic_structure.py \
  --input examples/data/simple_test.fasta \
  --output results/basic \
  --device cuda:0
```

**Parameters:**
- `--input, -i`: FASTA file with sequences (required)
- `--output, -o`: Output directory (default: results/)
- `--device`: Compute device (default: cuda:0)
- `--config`: Configuration file (optional)

#### MSA-Enhanced Prediction

```bash
# With local MSA files
python scripts/predict_with_msas.py \
  --input examples/data/sample.fasta \
  --msa-dir examples/data/ \
  --output results/msa

# With online MSA server (requires internet)
python scripts/predict_with_msas.py \
  --input examples/data/sample.fasta \
  --use-msa-server \
  --output results/msa_server
```

**Parameters:**
- `--input, -i`: FASTA file with sequences (required)
- `--output, -o`: Output directory (default: results/)
- `--msa-dir`: Directory with .aligned.pqt MSA files
- `--use-msa-server`: Use online MSA database
- `--device`: Compute device (default: cuda:0)

#### Batch Processing

```bash
# Sequential processing
python scripts/predict_batch_structures.py \
  --input-dir examples/data/batch_test \
  --output-dir results/batch

# Parallel processing (multiple GPUs)
python scripts/predict_batch_structures.py \
  --input-dir examples/data/batch_test \
  --output-dir results/batch \
  --parallel \
  --max-workers 2
```

**Parameters:**
- `--input-dir`: Directory containing FASTA files (required)
- `--output-dir`: Output directory (required)
- `--parallel`: Enable parallel processing
- `--max-workers`: Number of parallel workers (default: 2)
- `--file-pattern`: File pattern to match (default: *.fasta)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name chai-lab
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add chai-lab -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "chai-lab": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/chai1_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/chai1_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from chai-lab?
```

#### Basic Usage
```
Use validate_fasta_file with input_file "examples/data/sample.fasta"
```

#### Structure Prediction
```
Use submit_basic_prediction with input_file "examples/data/simple_test.fasta" and output_dir "results/my_prediction"
```

#### Job Management
```
Use get_job_status with job_id "abc123"
Use get_job_log with job_id "abc123" and tail 50
```

#### Batch Processing
```
Use submit_batch_prediction with input_dir "examples/data/batch_test" and parallel True
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sample.fasta` | Reference a specific FASTA file |
| `@configs/default_config.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "chai-lab": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/chai1_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/chai1_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use validate_fasta_file with input_file "examples/data/sample.fasta"
```

---

## Available Tools

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check status of submitted job |
| `get_job_result` | Get completed job results |
| `get_job_log` | View job execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all submitted jobs |

### Quick Operations (Sync API)

These tools return results immediately (< 3 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `validate_fasta_file` | Validate FASTA format and get sequence info | `input_file` |
| `predict_small_peptide` | Sync prediction for peptides ≤20 AA | `sequence`, `max_length`, `device`, `output_file` |
| `analyze_sequence_composition` | Analyze amino acid composition | `input_file` |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 3 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_basic_prediction` | Basic structure prediction | `input_file`, `output_dir`, `device`, model params |
| `submit_msa_prediction` | MSA-enhanced prediction | `input_file`, `output_dir`, `msa_directory`, `use_msa_server` |
| `submit_batch_prediction` | Batch processing | `input_dir`, `output_dir`, `parallel`, `max_workers` |

---

## Examples

### Example 1: Quick Validation and Analysis

**Goal:** Validate and analyze a FASTA file before prediction

**Using Script:**
```bash
python scripts/predict_basic_structure.py --help
```

**Using MCP (in Claude Code):**
```
Use validate_fasta_file with input_file "examples/data/sample.fasta"
Use analyze_sequence_composition with input_file "examples/data/sample.fasta"
```

**Expected Output:**
- Validation status and sequence statistics
- Amino acid composition analysis
- Prediction complexity assessment
- Recommended API (sync vs submit)

### Example 2: Small Peptide Prediction

**Goal:** Quick prediction for a small peptide

**Using Script:**
```bash
python scripts/predict_basic_structure.py \
  --input examples/data/simple_test.fasta \
  --output results/peptide/
```

**Using MCP (in Claude Code):**
```
Use predict_small_peptide with sequence "GAAL"
```

**Expected Output:**
- Immediate structure prediction (30 sec - 2 min)
- Confidence scores and metrics
- CIF structure file

### Example 3: Full Structure Prediction with Job Tracking

**Goal:** Predict structure for larger protein with progress monitoring

**Using Script:**
```bash
python scripts/predict_basic_structure.py \
  --input examples/data/sample.fasta \
  --output results/full_prediction/
```

**Using MCP (in Claude Code):**
```
Submit basic structure prediction for examples/data/sample.fasta

Then check job status periodically:
Use get_job_status with job_id "returned_job_id"
Use get_job_log with job_id "returned_job_id"

When completed:
Use get_job_result with job_id "returned_job_id"
```

### Example 4: MSA-Enhanced Prediction

**Goal:** Use evolutionary information for improved accuracy

**Using Script:**
```bash
python scripts/predict_with_msas.py \
  --input examples/data/sample.fasta \
  --use-msa-server \
  --output results/msa_enhanced/
```

**Using MCP (in Claude Code):**
```
Use submit_msa_prediction with:
- input_file "examples/data/sample.fasta"
- use_msa_server True
- output_dir "results/msa_enhanced"
```

### Example 5: Batch Processing

**Goal:** Process multiple files in parallel

**Using Script:**
```bash
python scripts/predict_batch_structures.py \
  --input-dir examples/data/batch_test \
  --output-dir results/batch \
  --parallel \
  --max-workers 2
```

**Using MCP (in Claude Code):**
```
Use submit_batch_prediction with:
- input_dir "examples/data/batch_test"
- output_dir "results/batch"
- parallel True
- max_workers 2
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `simple_test.fasta` | 4 AA peptide for quick testing | `predict_small_peptide` |
| `sample.fasta` | Multi-protein complex example | `submit_basic_prediction` |
| `1ac5.fasta` | Protein with glycan modifications | MSA prediction tools |
| `8cyo.fasta` | Protein-ligand complex | Any prediction tool |
| `protein_ligand.fasta` | Antibody with small molecule | Any prediction tool |
| `*.aligned.pqt` | Pre-computed MSA files | `submit_msa_prediction` |
| `batch_test/` | Multiple FASTA files | `submit_batch_prediction` |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Key Parameters |
|--------|-------------|----------------|
| `default_config.json` | Base settings for all predictions | model params, device, validation |
| `basic_prediction_config.json` | Basic prediction settings | recycles=3, timesteps=200 |
| `msa_prediction_config.json` | MSA prediction settings | MSA server options |
| `batch_prediction_config.json` | Batch processing settings | parallel options, worker limits |

### Config Example

```json
{
  "model": {
    "num_trunk_recycles": 3,
    "num_diffn_timesteps": 200,
    "seed": 42
  },
  "computation": {
    "device": "cuda:0",
    "fallback_device": "cpu"
  },
  "output": {
    "format": "cif",
    "include_scores": true
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install chai_lab==0.6.1 fastmcp loguru
```

**Problem:** Import errors
```bash
# Verify installation
python -c "import chai_lab; print('chai_lab available')"
python -c "from src.server import mcp; print('MCP server available')"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove chai-lab
claude mcp add chai-lab -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
mamba run -p ./env python -c "
import sys; sys.path.insert(0, 'src')
from server import mcp
print('Server name:', mcp.name)
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log
cat jobs/<job_id>/job.log
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

**Problem:** Out of GPU memory
```bash
# Reduce model parameters in config
{
  "num_trunk_recycles": 1,
  "num_diffn_timesteps": 50
}
```

### File Access Issues

**Problem:** File not found errors
```bash
# Use absolute paths
ls -la examples/data/sample.fasta

# Check permissions
chmod 644 examples/data/*.fasta
```

**Problem:** Output directory creation fails
```bash
# Create directory manually
mkdir -p results/test_output

# Check write permissions
touch results/test_output/test_file
```

---

## Performance Guidelines

### Tool Selection Guide

| Sequence Length | Recommended Tool | Expected Runtime |
|----------------|------------------|------------------|
| ≤ 20 AA | `predict_small_peptide` | 30 sec - 2 min |
| 21-100 AA | `submit_basic_prediction` | 3-10 min |
| 101-500 AA | `submit_basic_prediction` | 10-30 min |
| > 500 AA | `submit_msa_prediction` | 30+ min |
| Multiple files | `submit_batch_prediction` | Hours |

### Resource Usage

- **GPU Memory**: ~4-8GB for typical proteins
- **CPU Memory**: ~8-16GB during processing
- **Disk Space**: ~1-5MB per prediction output
- **Network**: Required for MSA server access

### Optimization Tips

1. **Validation First**: Always use `validate_fasta_file` before submission
2. **Sequence Analysis**: Use `analyze_sequence_composition` to assess complexity
3. **Batch Processing**: Use `submit_batch_prediction` for multiple files
4. **Parallel Processing**: Enable parallel=True for batch jobs with multiple GPUs
5. **Resource Management**: Monitor jobs with `get_job_status` and `get_job_log`

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test server startup
fastmcp dev src/server.py

# Test script functionality
python scripts/predict_basic_structure.py --help
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py
```

---

## License

Based on [chai-lab](https://github.com/chaidiscovery/chai-lab) by Chai Discovery.

## Credits

Based on [Chai Discovery's chai-lab repository](https://github.com/chaidiscovery/chai-lab)