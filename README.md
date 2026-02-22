# Chai-1 MCP Server

**Protein structure prediction using the Chai-1 model via Docker**

An MCP (Model Context Protocol) server for Chai-1 structure prediction with 6 core tools:
- Predict structures for small peptides (synchronous, instant results)
- Submit basic structure predictions from FASTA sequences
- Submit MSA-enhanced predictions for improved accuracy
- Batch process multiple FASTA files
- Monitor and retrieve job results
- Validate FASTA files before submission

## Quick Start with Docker

### Approach 1: Pull Pre-built Image from GitHub

The fastest way to get started. A pre-built Docker image is automatically published to GitHub Container Registry on every release.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/chai1_mcp:latest

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add chai1 -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` ghcr.io/macromnex/chai1_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support (`nvidia-docker` or Docker with NVIDIA runtime)
- Claude Code installed

That's it! The Chai-1 MCP server is now available in Claude Code.

---

### Approach 2: Build Docker Image Locally

Build the image yourself and install it into Claude Code. Useful for customization or offline environments.

```bash
# Clone the repository
git clone https://github.com/MacromNex/chai1_mcp.git
cd chai1_mcp

# Build the Docker image
docker build -t chai1_mcp:latest .

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add chai1 -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` chai1_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support
- Claude Code installed
- Git (to clone the repository)

**About the Docker Flags:**
- `-i` — Interactive mode for Claude Code
- `--rm` — Automatically remove container after exit
- `` --user `id -u`:`id -g` `` — Runs the container as your current user, so output files are owned by you (not root)
- `--gpus all` — Grants access to all available GPUs
- `--ipc=host` — Uses host IPC namespace for better performance
- `-v` — Mounts your project directory so the container can access your data

---

## Verify Installation

After adding the MCP server, you can verify it's working:

```bash
# List registered MCP servers
claude mcp list

# You should see 'chai1' in the output
```

In Claude Code, you can now use all 6 Chai-1 tools:
- `predict_small_peptide`
- `submit_basic_prediction`
- `submit_msa_prediction`
- `submit_batch_prediction`
- `get_job_status`
- `get_job_result`

---

## Next Steps

- **Detailed documentation**: See [detail.md](detail.md) for comprehensive guides on:
  - Available MCP tools and parameters
  - Local Python environment setup (alternative to Docker)
  - Example workflows and use cases
  - MSA server configuration
  - Configuration file format

---

## Usage Examples

Once registered, you can use the Chai-1 tools directly in Claude Code. Here are some common workflows:

### Example 1: Quick Peptide Prediction

```
I have a short peptide sequence "GAAKLKKTFR". Can you predict its structure using predict_small_peptide and save the result to /path/to/output/?
```

### Example 2: Full Protein Structure Prediction

```
I have a protein FASTA file at /path/to/protein.fasta. Can you submit a basic structure prediction using submit_basic_prediction with output saved to /path/to/results/, then monitor the job until it completes and retrieve the final structure?
```

### Example 3: MSA-Enhanced Prediction

```
I want high-accuracy structure prediction for my protein at /path/to/protein.fasta. Can you use submit_msa_prediction with use_msa_server set to True to include evolutionary information? Save results to /path/to/msa_results/.
```

---

## Troubleshooting

**Docker not found?**
```bash
docker --version  # Install Docker if missing
```

**GPU not accessible?**
- Ensure NVIDIA Docker runtime is installed
- Check with `docker run --gpus all ubuntu nvidia-smi`

**Claude Code not found?**
```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

---

## License

Based on [chai-lab](https://github.com/chaidiscovery/chai-lab) by Chai Discovery
