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

The chai-lab MCP provides access to the Chai-1 protein structure prediction model through both standalone Python scripts and an MCP server for integration with AI assistants like Claude Code and Gemini CLI. It enables prediction of protein, DNA, RNA, and ligand structures from FASTA sequences with optional Multiple Sequence Alignment (MSA) enhancement for improved accuracy.

### Features
- **Basic Structure Prediction**: Fast prediction from FASTA sequences without MSAs
- **MSA-Enhanced Prediction**: Higher accuracy predictions using evolutionary information
- **Batch Processing**: High-throughput processing of multiple FASTA files
- **Job Management**: Background processing with progress tracking for long-running tasks
- **Validation Tools**: Pre-flight validation and sequence analysis
- **Multiple Interfaces**: Standalone scripts, MCP server, and AI assistant integration

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   ├── server.py           # MCP server with 11 tools
│   └── jobs/               # Job management system
├── scripts/
│   ├── predict_basic_structure.py      # Basic structure prediction
│   ├── predict_with_msas.py           # MSA-enhanced prediction
│   ├── predict_batch_structures.py    # Batch processing
│   └── lib/                # Shared utilities (io.py, utils.py)
├── examples/
│   └── data/               # Demo data (FASTA files, MSA files)
├── configs/                # Configuration files
└── repo/                   # Original chai-lab repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- CUDA-capable GPU (recommended, CPU fallback available)
- 8GB+ GPU memory for typical proteins

### Step 1: Create Environment

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/chai1_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install MCP dependencies
pip install fastmcp loguru
```

### Step 3: Verify Installation

```bash
# Test imports
python -c "from src.server import mcp; print(f'Found {len(mcp.list_tools())} tools')"
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/predict_basic_structure.py` | Basic structure prediction from FASTA | See below |
| `scripts/predict_with_msas.py` | MSA-enhanced prediction with evolutionary info | See below |
| `scripts/predict_batch_structures.py` | Batch processing of multiple FASTA files | See below |

### Script Examples

#### Basic Structure Prediction

```bash
# Activate environment
mamba activate ./env

# Run basic prediction
python scripts/predict_basic_structure.py \
  --input examples/data/sample.fasta \
  --output results/basic \
  --config configs/basic_prediction_config.json
```

**Parameters:**
- `--input, -i`: Path to FASTA file with protein/DNA/RNA sequences (required)
- `--output, -o`: Output directory for results (default: results/)
- `--config, -c`: Configuration file path (optional)
- `--device`: Computing device (cuda:0, cuda:1, cpu)

#### MSA-Enhanced Prediction

```bash
# With local MSA files
python scripts/predict_with_msas.py \
  --input examples/data/sample.fasta \
  --msa-dir examples/data/ \
  --output results/msa

# With online MSA server
python scripts/predict_with_msas.py \
  --input examples/data/sample.fasta \
  --use-msa-server \
  --output results/msa_server
```

#### Batch Processing

```bash
# Sequential processing
python scripts/predict_batch_structures.py \
  --input-dir examples/data/batch_test \
  --output-dir results/batch

# Parallel processing (2 workers)
python scripts/predict_batch_structures.py \
  --input-dir examples/data/batch_test \
  --output-dir results/batch \
  --parallel \
  --max-workers 2
```

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
Use validate_fasta_file with input_file @examples/data/sample.fasta
```

#### Structure Prediction
```
Use submit_basic_prediction with input_file @examples/data/sample.fasta and output_dir "results/my_prediction"
```

#### With Configuration
```
Run predict_basic_structure on @examples/data/sample.fasta using config @configs/basic_prediction_config.json
```

#### Long-Running Tasks (Submit API)
```
Submit basic structure prediction for @examples/data/sample.fasta
Then check the job status
```

#### Batch Processing
```
Submit batch processing for all FASTA files in @examples/data/batch_test with parallel processing enabled
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
> Use validate_fasta_file with file examples/data/sample.fasta
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 3 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `validate_fasta_file` | Validate FASTA format and contents | `input_file` |
| `predict_small_peptide` | Predict structure for sequences ≤20 AA | `sequence`, `max_length`, `device`, `output_file` |
| `analyze_sequence_composition` | Analyze amino acid composition | `input_file` |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 3 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_basic_prediction` | Basic structure prediction | `input_file`, `output_dir`, `device`, `job_name`, etc. |
| `submit_msa_prediction` | MSA-enhanced prediction | `input_file`, `msa_directory`, `use_msa_server`, etc. |
| `submit_batch_prediction` | Batch processing of multiple files | `input_dir`, `parallel`, `max_workers`, etc. |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: Basic Protein Structure Prediction

**Goal:** Predict structure of a multi-protein complex with ligand

**Using Script:**
```bash
python scripts/predict_basic_structure.py \
  --input examples/data/sample.fasta \
  --output results/example1/
```

**Using MCP (in Claude Code):**
```
Use submit_basic_prediction to process @examples/data/sample.fasta and save results to results/example1/
```

**Expected Output:**
- Structure files in mmCIF format
- Confidence scores and metrics
- Prediction logs and metadata

### Example 2: MSA-Enhanced Prediction

**Goal:** Higher accuracy prediction using evolutionary information

**Using Script:**
```bash
python scripts/predict_with_msas.py \
  --input examples/data/sample.fasta \
  --use-msa-server \
  --output results/msa_enhanced/
```

**Using MCP (in Claude Code):**
```
Run submit_msa_prediction on @examples/data/sample.fasta with use_msa_server True
```

### Example 3: Batch Processing

**Goal:** Process multiple files at once

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
Submit batch processing for all FASTA files in @examples/data/batch_test with parallel True and max_workers 2
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `sample.fasta` | Multi-protein complex with fatty acid ligand | Basic prediction, MSA prediction |
| `simple_test.fasta` | Simple protein for testing | Small peptide prediction |
| `1ac5.fasta` | Sample protein structure | Basic prediction |
| `8cyo.fasta` | Another sample protein | Basic prediction |
| `protein_ligand.fasta` | Protein-ligand complex | Basic prediction |
| `*.aligned.pqt` | Pre-computed MSA files | MSA-enhanced prediction |
| `batch_test/` | Multiple FASTA files | Batch processing |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `default_config.json` | Base configuration for all tools | model settings, computation, output |
| `basic_prediction_config.json` | Basic prediction settings | num_trunk_recycles, device, output format |
| `msa_prediction_config.json` | MSA-specific settings | MSA server URLs, template settings |
| `batch_prediction_config.json` | Batch processing settings | parallel workers, resource limits |

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
    "include_scores": true,
    "include_metadata": true
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
pip install -r requirements.txt
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from src.server import mcp"
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
python -c "
from src.server import mcp
print(list(mcp.list_tools().keys()))
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

### Performance Issues

**Problem:** Out of GPU memory
- Reduce `num_trunk_recycles` from 3 to 1
- Reduce `num_diffn_timesteps` from 200 to 100
- Use CPU device as fallback

**Problem:** Slow predictions
- Ensure GPU is available and CUDA drivers are installed
- Use parallel processing for batch jobs
- Consider using MSA server for better accuracy vs speed tradeoff

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test server startup
python -c "from src.server import mcp; print('Server OK')"

# Test script functionality
python scripts/validate_use_cases.py
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py
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

### Resource Requirements

- **GPU Memory**: 4-8GB for typical proteins
- **RAM**: 8-16GB during processing
- **Disk Space**: 1-5MB per prediction output
- **Network**: Required for MSA server access

---

## License

Based on the original chai-lab repository from Chai Discovery.

## Credits

Based on [chai-lab](https://github.com/chaidiscovery/chai-lab) by Chai Discovery.

The MCP server implementation provides seamless integration with AI assistants while maintaining compatibility with standalone usage patterns.