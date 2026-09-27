#!/usr/bin/env python3
"""
Direct test of MCP tools bypassing FastMCP wrapper.
"""

import sys
import json
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_validate_fasta_file():
    """Test FASTA validation function directly."""
    print("\n=== Testing validate_fasta_file ===")

    # Import directly from server module
    from server import validate_fasta_file

    try:
        # Test with valid file
        result = validate_fasta_file("examples/data/sample.fasta")
        print(f"✓ Valid file test: {json.dumps(result, indent=2)}")

        # Test with invalid file
        result = validate_fasta_file("/nonexistent/file.fasta")
        print(f"✓ Invalid file test: {json.dumps(result, indent=2)}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_analyze_sequence_composition():
    """Test sequence composition analysis."""
    print("\n=== Testing analyze_sequence_composition ===")

    from server import analyze_sequence_composition

    try:
        result = analyze_sequence_composition("examples/data/sample.fasta")
        print(f"✓ Analysis result: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_predict_small_peptide():
    """Test small peptide prediction."""
    print("\n=== Testing predict_small_peptide ===")

    from server import predict_small_peptide

    try:
        # Test with very small peptide
        result = predict_small_peptide("GAAL", max_length=20)
        print(f"✓ Small peptide result: {json.dumps(result, indent=2)}")

        # Test with sequence too long
        result = predict_small_peptide("A" * 50, max_length=20)
        print(f"✓ Long sequence result: {json.dumps(result, indent=2)}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_job_management():
    """Test job management functions."""
    print("\n=== Testing Job Management ===")

    from server import list_jobs, submit_basic_prediction, get_job_status

    try:
        # Test list jobs
        result = list_jobs()
        print(f"✓ List jobs: {json.dumps(result, indent=2)}")

        # Test submit a basic prediction
        result = submit_basic_prediction(
            input_file="examples/data/sample.fasta",
            output_dir="results/test_submit",
            job_name="direct_test"
        )
        print(f"✓ Submit job: {json.dumps(result, indent=2)}")

        job_id = result.get("job_id")
        if job_id:
            # Test get job status
            status = get_job_status(job_id)
            print(f"✓ Job status: {json.dumps(status, indent=2)}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    print("Starting direct tool testing...")

    tests = [
        test_validate_fasta_file,
        test_analyze_sequence_composition,
        test_predict_small_peptide,
        test_job_management
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print(f"\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()