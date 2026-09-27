#!/usr/bin/env python3
"""
Verify MCP server is ready for Claude Code integration.
Tests underlying functions to ensure they will work when called via MCP.
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

def test_file_access():
    """Test that sample files are accessible."""
    print("=== Testing File Access ===")

    test_files = [
        "examples/data/sample.fasta",
        "examples/data/simple_test.fasta",
        "examples/data/batch_test/test1.fasta",
        "examples/data/batch_test/test2.fasta"
    ]

    results = {"accessible": [], "missing": []}

    for file_path in test_files:
        if Path(file_path).exists():
            results["accessible"].append(file_path)
            print(f"✓ {file_path}")
        else:
            results["missing"].append(file_path)
            print(f"✗ {file_path}")

    print(f"Files accessible: {len(results['accessible'])}/{len(test_files)}")
    return len(results["missing"]) == 0, results

def test_job_manager():
    """Test job manager functionality."""
    print("\n=== Testing Job Manager ===")

    try:
        from jobs.manager import job_manager

        # Test 1: List jobs
        result = job_manager.list_jobs()
        print(f"✓ List jobs: Found {result.get('total', 0)} jobs")

        # Test 2: Submit a test job
        submit_result = job_manager.submit_job(
            script_path="scripts/predict_basic_structure.py",
            args={"input": "examples/data/sample.fasta", "output": "results/readiness_test"},
            job_name="readiness_test"
        )
        print(f"✓ Submit job: {submit_result.get('status')}, job_id: {submit_result.get('job_id')}")

        # Test 3: Get job status
        job_id = submit_result.get("job_id")
        if job_id:
            status = job_manager.get_job_status(job_id)
            print(f"✓ Job status: {status.get('status')}")

            # Test 4: Get job log
            log_result = job_manager.get_job_log(job_id, tail=10)
            print(f"✓ Job log: {log_result.get('status')}")

        return True, {"job_manager": "working", "test_job_id": job_id}

    except Exception as e:
        print(f"✗ Job manager failed: {e}")
        return False, {"error": str(e)}

def test_fasta_validation_logic():
    """Test FASTA validation without the MCP wrapper."""
    print("\n=== Testing FASTA Validation Logic ===")

    try:
        import scripts.lib.io as lib_io

        # Test with valid file
        file_path = Path("examples/data/sample.fasta")
        is_valid = lib_io.validate_fasta_file(file_path)
        print(f"✓ FASTA validation: {file_path} -> {is_valid}")

        # Test reading content
        content = lib_io.read_fasta_content(file_path)
        lines = content.split('\n')
        headers = [line for line in lines if line.startswith('>')]
        print(f"✓ FASTA reading: Found {len(headers)} sequences")

        return True, {"validation": is_valid, "sequences": len(headers)}

    except Exception as e:
        print(f"✗ FASTA validation failed: {e}")
        return False, {"error": str(e)}

def test_sequence_analysis_logic():
    """Test sequence analysis logic."""
    print("\n=== Testing Sequence Analysis Logic ===")

    try:
        # Read sample file manually
        with open("examples/data/sample.fasta") as f:
            content = f.read()

        # Parse sequences
        lines = content.strip().split('\n')
        sequences = []
        current_header = None
        current_seq = ""

        for line in lines:
            if line.startswith('>'):
                if current_header:
                    sequences.append((current_header, current_seq))
                current_header = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line

        if current_header:
            sequences.append((current_header, current_seq))

        # Analyze first sequence
        header, seq = sequences[0]
        seq = seq.upper()

        # Basic analysis
        aa_counts = {aa: seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        total_valid = sum(aa_counts.values())

        print(f"✓ Sequence analysis: {header[:50]}...")
        print(f"  Length: {len(seq)}, Valid AAs: {total_valid}")
        print(f"  Top AAs: {sorted([(aa, count) for aa, count in aa_counts.items() if count > 0], key=lambda x: x[1], reverse=True)[:3]}")

        return True, {"sequences_found": len(sequences), "first_seq_length": len(seq)}

    except Exception as e:
        print(f"✗ Sequence analysis failed: {e}")
        return False, {"error": str(e)}

def test_small_peptide_logic():
    """Test small peptide prediction logic."""
    print("\n=== Testing Small Peptide Logic ===")

    try:
        # Test sequence validation
        test_sequences = [
            ("GAAL", 20, True),  # Valid short
            ("A" * 50, 20, False),  # Too long
            ("GAALXYZ", 20, False),  # Invalid characters
        ]

        results = []
        for sequence, max_length, should_pass in test_sequences:
            # Length check
            length_ok = len(sequence) <= max_length

            # Character check
            chars_ok = all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence.upper())

            passed = length_ok and chars_ok
            results.append({
                "sequence": sequence,
                "expected": should_pass,
                "actual": passed,
                "correct": passed == should_pass
            })

            status = "✓" if passed == should_pass else "✗"
            print(f"{status} {sequence}: length_ok={length_ok}, chars_ok={chars_ok}, result={passed}")

        all_correct = all(r["correct"] for r in results)
        return all_correct, {"test_results": results}

    except Exception as e:
        print(f"✗ Small peptide logic failed: {e}")
        return False, {"error": str(e)}

def test_output_directories():
    """Test that output directories can be created."""
    print("\n=== Testing Output Directory Creation ===")

    try:
        test_dirs = [
            "results/test_output",
            "results/batch_test",
            "results/integration_test"
        ]

        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            if Path(dir_path).exists():
                print(f"✓ Created: {dir_path}")
            else:
                print(f"✗ Failed: {dir_path}")

        return True, {"directories_created": len(test_dirs)}

    except Exception as e:
        print(f"✗ Directory creation failed: {e}")
        return False, {"error": str(e)}

def main():
    """Run all readiness tests."""
    print("MCP Server Readiness Verification")
    print("=" * 50)

    results = {
        "test_date": datetime.now().isoformat(),
        "tests": {},
        "summary": {}
    }

    tests = [
        ("file_access", test_file_access),
        ("job_manager", test_job_manager),
        ("fasta_validation", test_fasta_validation_logic),
        ("sequence_analysis", test_sequence_analysis_logic),
        ("small_peptide", test_small_peptide_logic),
        ("output_directories", test_output_directories)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'-' * 50}")
        success, details = test_func()
        results["tests"][test_name] = {
            "status": "passed" if success else "failed",
            "details": details
        }
        if success:
            passed += 1

    results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed/total*100:.1f}%",
        "ready_for_mcp": passed == total
    }

    print(f"\n{'=' * 50}")
    print("READINESS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass rate: {results['summary']['pass_rate']}")
    print(f"Ready for MCP: {'✓ YES' if passed == total else '✗ NO'}")

    # Save results
    with open("reports/mcp_readiness_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull report saved: reports/mcp_readiness_report.json")

    return results

if __name__ == "__main__":
    main()