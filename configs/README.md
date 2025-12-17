# Configuration Files

This directory contains JSON configuration files for MCP scripts. These externalize parameters to make scripts more flexible and MCP-friendly.

## Configuration Files

| File | Purpose | Script |
|------|---------|--------|
| `default_config.json` | Base configuration for all scripts | All scripts |
| `basic_prediction_config.json` | Basic structure prediction settings | `predict_basic_structure.py` |
| `msa_prediction_config.json` | MSA-enhanced prediction settings | `predict_with_msas.py` |
| `batch_prediction_config.json` | Batch processing settings | `predict_batch_structures.py` |

## Configuration Structure

### Model Parameters
```json
{
  "model": {
    "num_trunk_recycles": 3,        // Number of model recycles (1-5)
    "num_diffn_timesteps": 200,     // Diffusion steps (50-500)
    "seed": 42,                     // Random seed for reproducibility
    "use_esm_embeddings": true      // Use ESM protein embeddings
  }
}
```

### Computation Settings
```json
{
  "computation": {
    "device": "cuda:0",           // Primary device (cuda:N or cpu)
    "fallback_device": "cpu",     // Fallback if primary fails
    "parallel": false,            // Enable parallel processing
    "max_workers": 2              // Number of parallel workers
  }
}
```

### MSA Configuration
```json
{
  "msa": {
    "use_msa_server": false,           // Use online MSA server
    "use_templates_server": false,     // Use template server
    "msa_server_url": null,            // Custom MSA server URL
    "local_msa_directory": null,       // Local MSA files directory
    "msa_file_extension": ".aligned.pqt"  // MSA file extension
  }
}
```

### Output Settings
```json
{
  "output": {
    "format": "cif",                 // Output format (cif/pdb)
    "include_scores": true,          // Include confidence scores
    "include_metadata": true,        // Include prediction metadata
    "include_msa_plots": true,       // Generate MSA visualizations
    "clean_output_dir": true         // Clean output directory first
  }
}
```

### Validation Rules
```json
{
  "validation": {
    "validate_fasta": true,          // Validate FASTA format
    "validate_msa_files": true,      // Validate MSA files exist
    "min_sequence_length": 1,        // Minimum sequence length
    "max_sequence_length": 10000,    // Maximum sequence length
    "max_batch_size": 100            // Maximum batch size
  }
}
```

## Usage

### In Scripts
```python
# Load default config and override with custom
config = load_json("configs/default_config.json")
custom_config = load_json("configs/my_custom.json")
final_config = merge_configs(config, custom_config)
```

### From Command Line
```bash
# Use specific config file
python scripts/predict_basic_structure.py --config configs/basic_prediction_config.json --input file.fasta

# Override specific parameters
python scripts/predict_basic_structure.py --device cpu --num-trunk-recycles 5 --input file.fasta
```

## Configuration Priority

Parameters are applied in this order (higher priority overrides lower):

1. **Default values** in script
2. **Configuration file** (--config)
3. **Command line arguments** (--device, --seed, etc.)

## Parameter Descriptions

### Model Parameters

- **num_trunk_recycles** (1-5): Number of model iterations. Higher = better quality but slower
- **num_diffn_timesteps** (50-500): Diffusion sampling steps. Higher = better quality but slower
- **seed**: Random seed for reproducible results
- **use_esm_embeddings**: Use ESM protein language model embeddings (recommended)

### Device Selection

- **cuda:0, cuda:1, etc.**: Use specific GPU
- **cpu**: Use CPU (slower but always available)
- **auto**: Auto-detect best available device

### MSA Options

- **use_msa_server**: Query online MSA databases (requires internet)
- **use_templates_server**: Use structural templates with MSA server
- **local_msa_directory**: Use pre-computed local MSA files (.aligned.pqt format)

### Performance Tuning

For **fast testing**:
```json
{
  "model": {
    "num_trunk_recycles": 1,
    "num_diffn_timesteps": 50
  }
}
```

For **high quality**:
```json
{
  "model": {
    "num_trunk_recycles": 5,
    "num_diffn_timesteps": 500
  }
}
```

For **batch processing**:
```json
{
  "computation": {
    "parallel": true,
    "max_workers": 4
  }
}
```

## Validation

All configuration files are validated at runtime:
- Parameter ranges checked
- File paths verified
- Device availability tested
- Invalid parameters cause clear error messages

## Custom Configurations

Create custom configurations by copying and modifying existing files:

```bash
# Create custom config
cp configs/basic_prediction_config.json configs/my_custom.json
# Edit my_custom.json as needed
python scripts/predict_basic_structure.py --config configs/my_custom.json --input file.fasta
```