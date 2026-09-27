#!/usr/bin/env python3
"""
Comprehensive test suite for chai-lab MCP server tools.
"""

import sys
import json
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the server
from server import mcp

class ToolTester:
    def __init__(self):
        self.results = {
            "test_date": None,
            "sync_tools": {},
            "submit_tools": {},
            "job_management": {},
            "issues": [],
            "summary": {}
        }

    def test_sync_tool(self, tool_name: str, **kwargs):
        """Test a synchronous tool."""
        print(f"\n=== Testing {tool_name} ===")
        try:
            # Get the tool function
            tool_func = getattr(sys.modules['server'], tool_name)
            result = tool_func(**kwargs)

            print(f"✓ {tool_name} succeeded")
            print(f"Result: {json.dumps(result, indent=2)}")

            self.results["sync_tools"][tool_name] = {
                "status": "passed",
                "input": kwargs,
                "output": result
            }
            return True

        except Exception as e:
            print(f"✗ {tool_name} failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")

            self.results["sync_tools"][tool_name] = {
                "status": "failed",
                "input": kwargs,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.results["issues"].append({
                "tool": tool_name,
                "type": "sync_tool_error",
                "error": str(e)
            })
            return False

    def test_submit_tool(self, tool_name: str, **kwargs):
        """Test a submit tool (returns job_id)."""
        print(f"\n=== Testing {tool_name} ===")
        try:
            tool_func = getattr(sys.modules['server'], tool_name)
            result = tool_func(**kwargs)

            print(f"✓ {tool_name} succeeded")
            print(f"Result: {json.dumps(result, indent=2)}")

            job_id = result.get("job_id")
            if job_id:
                print(f"Job submitted with ID: {job_id}")

            self.results["submit_tools"][tool_name] = {
                "status": "passed",
                "input": kwargs,
                "output": result
            }
            return result.get("job_id")

        except Exception as e:
            print(f"✗ {tool_name} failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")

            self.results["submit_tools"][tool_name] = {
                "status": "failed",
                "input": kwargs,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.results["issues"].append({
                "tool": tool_name,
                "type": "submit_tool_error",
                "error": str(e)
            })
            return None

    def test_job_management_tool(self, tool_name: str, **kwargs):
        """Test a job management tool."""
        print(f"\n=== Testing {tool_name} ===")
        try:
            tool_func = getattr(sys.modules['server'], tool_name)
            result = tool_func(**kwargs)

            print(f"✓ {tool_name} succeeded")
            print(f"Result: {json.dumps(result, indent=2)}")

            self.results["job_management"][tool_name] = {
                "status": "passed",
                "input": kwargs,
                "output": result
            }
            return True

        except Exception as e:
            print(f"✗ {tool_name} failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")

            self.results["job_management"][tool_name] = {
                "status": "failed",
                "input": kwargs,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.results["issues"].append({
                "tool": tool_name,
                "type": "job_mgmt_error",
                "error": str(e)
            })
            return False

    def run_all_tests(self):
        """Run comprehensive test suite."""
        from datetime import datetime
        self.results["test_date"] = datetime.now().isoformat()

        print("Starting comprehensive MCP tool testing...")

        # Test 1: FASTA validation
        print("\n" + "="*60)
        print("TESTING SYNC TOOLS")
        print("="*60)

        self.test_sync_tool("validate_fasta_file",
                           input_file="examples/data/sample.fasta")

        self.test_sync_tool("validate_fasta_file",
                           input_file="/nonexistent/file.fasta")  # Error case

        # Test 2: Sequence composition analysis
        self.test_sync_tool("analyze_sequence_composition",
                           input_file="examples/data/sample.fasta")

        # Test 3: Small peptide prediction (if sequence is small enough)
        self.test_sync_tool("predict_small_peptide",
                           sequence="MKLLILVAAAAALAVDAGAEE",
                           max_length=25,
                           output_file="results/test_peptide.pdb")

        self.test_sync_tool("predict_small_peptide",
                           sequence="A" * 50,  # Should fail - too long
                           max_length=20)

        # Test 4: Job management tools
        print("\n" + "="*60)
        print("TESTING JOB MANAGEMENT TOOLS")
        print("="*60)

        self.test_job_management_tool("list_jobs")

        # Test 5: Submit tools
        print("\n" + "="*60)
        print("TESTING SUBMIT TOOLS")
        print("="*60)

        job_id = self.test_submit_tool("submit_basic_prediction",
                                      input_file="examples/data/sample.fasta",
                                      output_dir="results/test_submit",
                                      job_name="test_basic_prediction")

        if job_id:
            # Test job status checking
            self.test_job_management_tool("get_job_status", job_id=job_id)
            self.test_job_management_tool("get_job_log", job_id=job_id, tail=20)

        # Test batch submission
        batch_job_id = self.test_submit_tool("submit_batch_prediction",
                                           input_dir="examples/data/batch_test",
                                           output_dir="results/test_batch",
                                           file_pattern="*.fasta",
                                           job_name="test_batch")

        if batch_job_id:
            self.test_job_management_tool("get_job_status", job_id=batch_job_id)

        # Generate summary
        self.generate_summary()
        return self.results

    def generate_summary(self):
        """Generate test summary."""
        sync_total = len(self.results["sync_tools"])
        sync_passed = sum(1 for t in self.results["sync_tools"].values() if t["status"] == "passed")

        submit_total = len(self.results["submit_tools"])
        submit_passed = sum(1 for t in self.results["submit_tools"].values() if t["status"] == "passed")

        job_total = len(self.results["job_management"])
        job_passed = sum(1 for t in self.results["job_management"].values() if t["status"] == "passed")

        total_tests = sync_total + submit_total + job_total
        total_passed = sync_passed + submit_passed + job_passed

        self.results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "pass_rate": f"{total_passed/total_tests*100:.1f}%" if total_tests > 0 else "N/A",
            "sync_tools": {"passed": sync_passed, "failed": sync_total - sync_passed},
            "submit_tools": {"passed": submit_passed, "failed": submit_total - submit_passed},
            "job_management": {"passed": job_passed, "failed": job_total - job_passed}
        }

        print(f"\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Pass rate: {self.results['summary']['pass_rate']}")
        print(f"Issues found: {len(self.results['issues'])}")

        for issue in self.results["issues"]:
            print(f"  - {issue['tool']}: {issue['error']}")

if __name__ == "__main__":
    tester = ToolTester()
    results = tester.run_all_tests()

    # Save results
    with open("reports/test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to: reports/test_results.json")