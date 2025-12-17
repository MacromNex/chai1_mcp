# Chai-1 MCP Examples

This directory contains use case scripts and demo data for the Chai-1 MCP server.

## Use Case Scripts

The following scripts demonstrate the main use cases for Chai-1 structure prediction:

### UC-001: Basic Structure Prediction
- **Script**: `use_case_1_basic_structure_prediction.py`
- **Description**: Basic structure prediction from FASTA sequence without MSAs
- **Complexity**: Simple
- **Priority**: High
- **Environment**: `./env`

**Usage:**
```bash
# Use sample data
python examples/use_case_1_basic_structure_prediction.py

# Predict from custom FASTA
python examples/use_case_1_basic_structure_prediction.py --input protein.fasta --output results/

# Use CPU instead of GPU
python examples/use_case_1_basic_structure_prediction.py --device cpu
```

**Input**: FASTA file with protein/DNA/RNA sequences and ligands (SMILES)
**Output**: CIF structure files with confidence scores

---

### UC-002: Structure Prediction with MSAs
- **Script**: `use_case_2_prediction_with_msas.py`
- **Description**: Structure prediction using MSAs (Multiple Sequence Alignments) for improved accuracy
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env`

**Usage:**
```bash
# Use online MSA server (requires internet)
python examples/use_case_2_prediction_with_msas.py --use-msa-server

# Use online MSA server with templates
python examples/use_case_2_prediction_with_msas.py --use-msa-server --use-templates-server

# Use local MSA files
python examples/use_case_2_prediction_with_msas.py --msa-dir examples/data/

# Custom FASTA with MSA server
python examples/use_case_2_prediction_with_msas.py --input protein.fasta --use-msa-server
```

**Input**: FASTA file + MSAs (local .pqt files or online MSA server)
**Output**: CIF structure files with improved accuracy

---

### UC-003: Batch Structure Prediction
- **Script**: `use_case_3_batch_prediction.py`
- **Description**: Batch processing of multiple FASTA files for high-throughput prediction
- **Complexity**: Medium
- **Priority**: Medium
- **Environment**: `./env`

**Usage:**
```bash
# Process sample data sequentially
python examples/use_case_3_batch_prediction.py

# Process directory of FASTA files
python examples/use_case_3_batch_prediction.py --input-dir fastas/ --output-dir batch_results/

# Parallel processing with multiple GPUs
python examples/use_case_3_batch_prediction.py --parallel --max-workers 2

# Batch processing with MSAs
python examples/use_case_3_batch_prediction.py --use-msa-server
```

**Input**: Directory of FASTA files
**Output**: Individual prediction results + batch summary report

---

## Demo Data

### Sample FASTA Data
The scripts automatically generate sample FASTA files in `data/` when run without input parameters:
- `sample_basic.fasta` - Basic prediction sample
- `sample_msa.fasta` - MSA prediction sample
- `batch_samples/` - Multiple files for batch processing

### MSA Files
Pre-computed MSA files in `.aligned.pqt` format:
- `703adc2c74b8d7e613549b6efcf37126da7963522dc33852ad3c691eef1da06f.aligned.pqt`
- `952a89ff052afbe8cd1656a317de8a4aa2457d6d73f50d228961bb84efd17e02.aligned.pqt`

These files contain Multiple Sequence Alignments that can be used with UC-002 for improved prediction accuracy.

## FASTA Format Guidelines

Chai-1 uses a specific FASTA format with entity type annotations:

```fasta
>protein|name=my-protein
AGSHSMRYFSTSVSRPGRGEPRFI...

>ligand|name=my-ligand
CCO  # SMILES notation for small molecules

>dna|name=my-dna
ATCGATCGATCG

>rna|name=my-rna
AUCGAUCGAUCG
```

### Entity Types:
- `protein` - Amino acid sequences
- `ligand` - Small molecules in SMILES format
- `dna` - DNA sequences
- `rna` - RNA sequences

### Modified Residues:
Use the format `AAA(MOD)AAA` for modified amino acids:
```fasta
>protein|name=modified-protein
AGSHSMRYF(SEP)TSVSRPGRGEPRFI
```

## System Requirements

- **GPU**: CUDA-compatible GPU recommended (RTX 4090, A100, H100, etc.)
- **Memory**: 8GB+ GPU memory for typical proteins
- **CPU**: Works but significantly slower
- **Internet**: Required for MSA server usage (UC-002)

## Output Files

All prediction scripts generate:
- `*.cif` - Structure files in mmCIF format
- `scores.model_idx_*.npz` - Confidence scores and metrics
- `*.pdb` - Structure files in PDB format (if requested)
- Summary logs with aggregate scores

## Error Handling

Common issues and solutions:

1. **CUDA out of memory**: Use smaller proteins or CPU device
2. **MSA server timeout**: Try again or use local MSA files
3. **Import errors**: Ensure chai_lab is installed: `pip install chai_lab==0.6.1`
4. **Permission errors**: Check file write permissions in output directories

## Next Steps

After running examples:
1. Try your own protein sequences
2. Experiment with different parameters (recycles, timesteps)
3. Use the results for downstream analysis
4. Convert to MCP server tools for programmatic access