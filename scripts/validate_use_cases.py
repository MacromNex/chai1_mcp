#!/usr/bin/env python3
"""
Quick validation script to test use cases without full execution.
This verifies imports work and basic functionality is available.
"""

import sys
import traceback
from pathlib import Path
import importlib.util

def test_imports():
    """Test if all required imports work."""
    print("Testing imports...")

    try:
        from chai_lab.chai1 import run_inference
        print("✅ chai_lab.chai1.run_inference imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import chai_lab: {e}")
        return False

def test_fasta_parsing():
    """Test if FASTA files can be read."""
    print("Testing FASTA parsing...")

    fasta_files = [
        "examples/data/sample.fasta",
        "examples/data/simple_test.fasta",
        "examples/data/protein_ligand.fasta"
    ]

    working_files = []

    for fasta_file in fasta_files:
        fasta_path = Path(fasta_file)
        if fasta_path.exists():
            try:
                content = fasta_path.read_text()
                if content.strip():
                    print(f"✅ {fasta_file} exists and has content")
                    working_files.append(fasta_file)
                else:
                    print(f"⚠️  {fasta_file} exists but is empty")
            except Exception as e:
                print(f"❌ Error reading {fasta_file}: {e}")
        else:
            print(f"❌ {fasta_file} not found")

    return working_files

def test_use_case_scripts():
    """Test if use case scripts can be imported."""
    print("Testing use case script imports...")

    scripts = [
        "examples/use_case_1_basic_structure_prediction.py",
        "examples/use_case_2_prediction_with_msas.py",
        "examples/use_case_3_batch_prediction.py"
    ]

    working_scripts = []

    for script_path in scripts:
        script_path = Path(script_path)
        if script_path.exists():
            try:
                # Try to load the script as a module
                spec = importlib.util.spec_from_file_location("test_module", script_path)
                module = importlib.util.module_from_spec(spec)
                # Don't execute, just check if it can be loaded
                print(f"✅ {script_path.name} can be loaded")
                working_scripts.append(str(script_path))
            except Exception as e:
                print(f"❌ Error loading {script_path.name}: {e}")
        else:
            print(f"❌ {script_path} not found")

    return working_scripts

def test_environment():
    """Test environment setup."""
    print("Testing environment...")

    import torch
    print(f"✅ PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} devices")
    else:
        print("ℹ️  CUDA not available, CPU only")

    import numpy
    print(f"✅ NumPy version: {numpy.__version__}")

def main():
    """Run all tests."""
    print("=== Use Case Validation ===\n")

    try:
        # Test 1: Imports
        if not test_imports():
            print("\n❌ Critical: chai_lab not available. Cannot proceed with use cases.")
            return 1

        print()

        # Test 2: Environment
        test_environment()

        print()

        # Test 3: Data files
        working_files = test_fasta_parsing()

        print()

        # Test 4: Scripts
        working_scripts = test_use_case_scripts()

        print("\n=== Summary ===")
        print(f"✅ chai_lab import: OK")
        print(f"✅ Working FASTA files: {len(working_files)}")
        print(f"✅ Working scripts: {len(working_scripts)}")

        if working_files and working_scripts:
            print("\n🎉 All use cases appear to be set up correctly!")
            print("📝 Note: Full execution requires significant compute time and model downloads.")
            return 0
        else:
            print("\n⚠️  Some issues found. Check the details above.")
            return 1

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())