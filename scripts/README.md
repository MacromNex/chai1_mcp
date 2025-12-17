# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `predict_basic_structure.py` | Basic structure prediction | Yes (chai_lab) | `configs/basic_prediction_config.json` |
| `predict_with_msas.py` | MSA-enhanced prediction | Yes (chai_lab) | `configs/msa_prediction_config.json` |
| `predict_batch_structures.py` | Batch prediction processing | Yes (chai_lab) | `configs/batch_prediction_config.json` |

## Usage

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Basic structure prediction
python scripts/predict_basic_structure.py --input examples/data/simple_test.fasta --output results/basic

# MSA-enhanced prediction with local MSA files
python scripts/predict_with_msas.py --input examples/data/simple_test.fasta --msa-dir examples/data/ --output results/msa

# MSA-enhanced prediction with online server
python scripts/predict_with_msas.py --input examples/data/simple_test.fasta --use-msa-server --output results/msa_server

# Batch prediction
python scripts/predict_batch_structures.py --input-dir examples/data/batch_test --output-dir results/batch

# With custom config
python scripts/predict_basic_structure.py --input FILE --output DIR --config configs/custom.json
```

## Shared Library

Common functions are in `scripts/lib/`:
- `io.py`: File loading/saving, FASTA validation
- `utils.py`: Configuration management, timing, device validation

## Configuration Files

All scripts support JSON configuration files in `configs/`:

- `default_config.json`: Base configuration
- `basic_prediction_config.json`: Basic prediction settings
- `msa_prediction_config.json`: MSA-specific settings
- `batch_prediction_config.json`: Batch processing settings

### Example Configuration

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

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from scripts.predict_basic_structure import run_predict_basic_structure

# In MCP tool:
@mcp.tool()
def predict_structure(input_file: str, output_dir: str = None):
    return run_predict_basic_structure(input_file, output_dir)
```

### Main Functions

- `run_predict_basic_structure()` - Basic prediction
- `run_predict_with_msas()` - MSA-enhanced prediction
- `run_predict_batch_structures()` - Batch processing

## Dependencies

### Essential (Required)
- `numpy` - Array operations
- `pathlib` - Path handling
- `argparse` - CLI interface

### Repo Required (Cannot be inlined)
- `chai_lab.chai1.run_inference` - Core Chai-1 model inference

### Lazy Loading
All heavy dependencies (chai_lab) are loaded lazily to minimize startup time and memory usage when not needed.

## Error Handling

All scripts include comprehensive error handling:
- Input validation
- Device validation
- Model loading errors
- File system errors
- Graceful fallbacks where possible

## Testing

Scripts have been tested with:
- Sample data from `examples/data/`
- Both CPU and GPU modes
- Config file overrides
- CLI parameter validation

## Performance Notes

- **GPU Acceleration**: Significantly faster than CPU
- **Model Downloads**: First run downloads ~3GB of models
- **Memory Usage**: 8GB+ GPU memory recommended
- **Parallel Processing**: Supported for batch processing with multiple GPUs