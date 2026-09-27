#!/usr/bin/env python3
"""
Integration test that properly tests MCP server using the protocol.
"""

import sys
import json
import asyncio
import traceback
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

async def test_mcp_tools():
    """Test MCP tools using proper FastMCP async API."""
    print("Starting MCP integration tests...")

    # Import the server
    from server import mcp

    results = {
        "test_date": None,
        "tools_tested": {},
        "issues": [],
        "summary": {}
    }

    try:
        # Test 1: List all available tools
        print("\n=== Testing: List Tools ===")
        tools = await mcp.get_tools()
        tool_names = list(tools.keys())
        print(f"✓ Found {len(tool_names)} tools:")
        for name in tool_names:
            print(f"  - {name}")

        # Test 2: Test sync tools
        print(f"\n=== Testing Sync Tools ===")

        # Test validate_fasta_file
        print(f"\n--- Testing validate_fasta_file ---")
        try:
            result = await mcp.call_tool("validate_fasta_file", {
                "input_file": "examples/data/sample.fasta"
            })
            print(f"✓ validate_fasta_file: {json.dumps(result, indent=2)}")
            results["tools_tested"]["validate_fasta_file"] = {"status": "passed", "result": result}
        except Exception as e:
            print(f"✗ validate_fasta_file failed: {e}")
            results["tools_tested"]["validate_fasta_file"] = {"status": "failed", "error": str(e)}
            results["issues"].append({"tool": "validate_fasta_file", "error": str(e)})

        # Test analyze_sequence_composition
        print(f"\n--- Testing analyze_sequence_composition ---")
        try:
            result = await mcp.call_tool("analyze_sequence_composition", {
                "input_file": "examples/data/sample.fasta"
            })
            print(f"✓ analyze_sequence_composition: {json.dumps(result, indent=2)}")
            results["tools_tested"]["analyze_sequence_composition"] = {"status": "passed", "result": result}
        except Exception as e:
            print(f"✗ analyze_sequence_composition failed: {e}")
            results["tools_tested"]["analyze_sequence_composition"] = {"status": "failed", "error": str(e)}
            results["issues"].append({"tool": "analyze_sequence_composition", "error": str(e)})

        # Test predict_small_peptide
        print(f"\n--- Testing predict_small_peptide ---")
        try:
            result = await mcp.call_tool("predict_small_peptide", {
                "sequence": "GAAL",
                "max_length": 20
            })
            print(f"✓ predict_small_peptide: {json.dumps(result, indent=2)}")
            results["tools_tested"]["predict_small_peptide"] = {"status": "passed", "result": result}
        except Exception as e:
            print(f"✗ predict_small_peptide failed: {e}")
            results["tools_tested"]["predict_small_peptide"] = {"status": "failed", "error": str(e)}
            results["issues"].append({"tool": "predict_small_peptide", "error": str(e)})

        # Test 3: Test job management tools
        print(f"\n=== Testing Job Management Tools ===")

        # Test list_jobs
        print(f"\n--- Testing list_jobs ---")
        try:
            result = await mcp.call_tool("list_jobs", {})
            print(f"✓ list_jobs: {json.dumps(result, indent=2)}")
            results["tools_tested"]["list_jobs"] = {"status": "passed", "result": result}
        except Exception as e:
            print(f"✗ list_jobs failed: {e}")
            results["tools_tested"]["list_jobs"] = {"status": "failed", "error": str(e)}
            results["issues"].append({"tool": "list_jobs", "error": str(e)})

        # Test 4: Test submit tools
        print(f"\n=== Testing Submit Tools ===")

        # Test submit_basic_prediction
        print(f"\n--- Testing submit_basic_prediction ---")
        try:
            result = await mcp.call_tool("submit_basic_prediction", {
                "input_file": "examples/data/sample.fasta",
                "output_dir": "results/test_integration",
                "job_name": "integration_test"
            })
            print(f"✓ submit_basic_prediction: {json.dumps(result, indent=2)}")
            results["tools_tested"]["submit_basic_prediction"] = {"status": "passed", "result": result}

            # Test getting job status for submitted job
            job_id = result.get("job_id")
            if job_id:
                print(f"\n--- Testing get_job_status for job {job_id} ---")
                try:
                    status_result = await mcp.call_tool("get_job_status", {"job_id": job_id})
                    print(f"✓ get_job_status: {json.dumps(status_result, indent=2)}")
                    results["tools_tested"]["get_job_status"] = {"status": "passed", "result": status_result}
                except Exception as e:
                    print(f"✗ get_job_status failed: {e}")
                    results["tools_tested"]["get_job_status"] = {"status": "failed", "error": str(e)}
                    results["issues"].append({"tool": "get_job_status", "error": str(e)})

        except Exception as e:
            print(f"✗ submit_basic_prediction failed: {e}")
            results["tools_tested"]["submit_basic_prediction"] = {"status": "failed", "error": str(e)}
            results["issues"].append({"tool": "submit_basic_prediction", "error": str(e)})

        # Generate summary
        total_tested = len(results["tools_tested"])
        total_passed = sum(1 for t in results["tools_tested"].values() if t["status"] == "passed")

        results["summary"] = {
            "total_tools_available": len(tool_names),
            "total_tested": total_tested,
            "total_passed": total_passed,
            "total_failed": total_tested - total_passed,
            "pass_rate": f"{total_passed/total_tested*100:.1f}%" if total_tested > 0 else "N/A",
            "issues_found": len(results["issues"])
        }

        print(f"\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Available tools: {len(tool_names)}")
        print(f"Tools tested: {total_tested}")
        print(f"Tests passed: {total_passed}")
        print(f"Tests failed: {total_tested - total_passed}")
        print(f"Pass rate: {results['summary']['pass_rate']}")
        print(f"Issues found: {len(results['issues'])}")

        if results["issues"]:
            print(f"\nIssues:")
            for issue in results["issues"]:
                print(f"  - {issue['tool']}: {issue['error']}")

        # Save results
        with open("reports/integration_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nFull results saved to: reports/integration_test_results.json")

        return results

    except Exception as e:
        print(f"✗ Critical error in testing: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main entry point."""
    # Run the async test
    return asyncio.run(test_mcp_tools())

if __name__ == "__main__":
    main()