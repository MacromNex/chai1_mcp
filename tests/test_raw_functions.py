#!/usr/bin/env python3
"""
Test the raw functions before they're wrapped by FastMCP.
"""

import sys
import json
import traceback
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

def test_validate_fasta_direct():
    """Test FASTA validation directly from the lib."""
    print("\n=== Testing FASTA validation (lib) ===")
    try:
        # Import the lib functions directly
        import scripts.lib.io as lib_io
        validate_fasta_file = lib_io.validate_fasta_file
        read_fasta_content = lib_io.read_fasta_content

        # Test validation
        is_valid = validate_fasta_file("examples/data/sample.fasta")
        print(f"✓ File validation: {is_valid}")

        # Test reading
        sequences = read_fasta_content("examples/data/sample.fasta")
        print(f"✓ Read {len(sequences)} sequences")

        for header, seq in sequences:
            print(f"  - {header}: {len(seq)} residues")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_job_manager():
    """Test job manager directly."""
    print("\n=== Testing Job Manager ===")
    try:
        from jobs.manager import job_manager

        # List jobs
        result = job_manager.list_jobs()
        print(f"✓ List jobs: {json.dumps(result, indent=2)}")

        # Submit a job
        result = job_manager.submit_job(
            script_path="scripts/predict_basic_structure.py",
            args={"input": "examples/data/sample.fasta", "output": "results/test_job"},
            job_name="test_job"
        )
        print(f"✓ Submit job: {json.dumps(result, indent=2)}")

        job_id = result.get("job_id")
        if job_id:
            status = job_manager.get_job_status(job_id)
            print(f"✓ Job status: {json.dumps(status, indent=2)}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_sequence_analysis():
    """Test sequence analysis manually."""
    print("\n=== Testing Sequence Analysis (manual) ===")
    try:
        import scripts.lib.io as lib_io
        read_fasta_content = lib_io.read_fasta_content

        sequences = read_fasta_content("examples/data/sample.fasta")
        analysis = []

        for header, seq in sequences:
            seq = seq.upper()
            seq_len = len(seq)

            # Amino acid composition
            aa_counts = {aa: seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
            aa_freq = {aa: count/seq_len for aa, count in aa_counts.items() if count > 0}

            # Basic properties
            hydrophobic = sum(aa_counts[aa] for aa in "AILVFWY") / seq_len
            charged = sum(aa_counts[aa] for aa in "DEKR") / seq_len
            polar = sum(aa_counts[aa] for aa in "STNQC") / seq_len

            analysis.append({
                "header": header,
                "length": seq_len,
                "top_amino_acids": dict(sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)[:5]),
                "properties": {
                    "hydrophobic_fraction": round(hydrophobic, 3),
                    "charged_fraction": round(charged, 3),
                    "polar_fraction": round(polar, 3)
                },
                "complexity": "simple" if seq_len < 50 else "moderate" if seq_len < 200 else "complex"
            })

        print(f"✓ Analyzed {len(analysis)} sequences:")
        for seq_analysis in analysis:
            print(f"  - {seq_analysis['header']}: {seq_analysis['length']} residues, {seq_analysis['complexity']} complexity")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_small_prediction_logic():
    """Test the logic of small prediction without actual prediction."""
    print("\n=== Testing Small Prediction Logic ===")
    try:
        # Test sequence validation
        sequence = "GAAL"
        max_length = 20

        if len(sequence) > max_length:
            print(f"✓ Length check: Sequence too long ({len(sequence)} > {max_length})")
            return True

        # Validate sequence characters
        if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence.upper()):
            print(f"✗ Invalid amino acid characters in sequence: {sequence}")
            return False

        print(f"✓ Sequence '{sequence}' passed validation (length: {len(sequence)}, max: {max_length})")

        # Test with invalid sequence
        invalid_seq = "GAALXYZ"
        if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in invalid_seq.upper()):
            print(f"✓ Invalid sequence '{invalid_seq}' correctly rejected")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing raw functions and libraries...")

    tests = [
        test_validate_fasta_direct,
        test_sequence_analysis,
        test_small_prediction_logic,
        test_job_manager
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print(f"\n" + "="*60)
    print(f"RAW FUNCTION TESTS: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print("="*60)

    return passed == total

if __name__ == "__main__":
    main()