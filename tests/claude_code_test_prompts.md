# Claude Code MCP Integration Test Prompts

This document contains test prompts to manually verify the chai-lab MCP server integration in Claude Code.

## Prerequisites

1. MCP server should be registered: `claude mcp list` should show `chai-lab: ✓ Connected`
2. Sample data files should be available in `examples/data/`
3. Run tests from the MCP directory: `/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/chai1_mcp/`

## Test 1: Tool Discovery

**Prompt:**
```
What MCP tools are available from chai-lab? Please list them with their descriptions.
```

**Expected Result:**
- Should list 11 tools
- Should include both sync and submit tools
- Should show job management tools

---

## Test 2: FASTA File Validation (Sync Tool)

**Prompt:**
```
Use the validate_fasta_file tool to check the file examples/data/sample.fasta
```

**Expected Result:**
```json
{
  "status": "success",
  "file": "examples/data/sample.fasta",
  "total_sequences": 4,
  "total_length": <number>,
  "sequences": [...],
  "estimated_runtime_minutes": <number>,
  "recommended_api": "sync|submit"
}
```

---

## Test 3: FASTA File Validation Error Handling

**Prompt:**
```
Use the validate_fasta_file tool with a non-existent file path: /fake/nonexistent.fasta
```

**Expected Result:**
```json
{
  "status": "error",
  "error": "File not found: /fake/nonexistent.fasta"
}
```

---

## Test 4: Sequence Composition Analysis

**Prompt:**
```
Use the analyze_sequence_composition tool to analyze examples/data/sample.fasta
```

**Expected Result:**
```json
{
  "status": "success",
  "file": "examples/data/sample.fasta",
  "sequences": [
    {
      "header": "protein|name=example-protein",
      "length": <number>,
      "composition": {...},
      "properties": {
        "hydrophobic_fraction": <number>,
        "charged_fraction": <number>,
        "polar_fraction": <number>
      },
      "prediction_complexity": "simple|moderate|complex"
    },
    ...
  ],
  "total_sequences": 4
}
```

---

## Test 5: Small Peptide Prediction

**Prompt:**
```
Use the predict_small_peptide tool with sequence "GAAL" and max_length 20
```

**Expected Result:**
- Should either succeed with prediction results or fail with dependency issues
- If it fails, error should be informative about missing dependencies

---

## Test 6: Small Peptide Validation (Too Long)

**Prompt:**
```
Use the predict_small_peptide tool with sequence "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY" and max_length 20
```

**Expected Result:**
```json
{
  "status": "error",
  "error": "Sequence too long (60 > 20). Use submit_basic_prediction for longer sequences."
}
```

---

## Test 7: List Jobs

**Prompt:**
```
Use the list_jobs tool to see all submitted jobs
```

**Expected Result:**
```json
{
  "status": "success",
  "jobs": [...],
  "total": <number>
}
```

---

## Test 8: Submit Basic Prediction

**Prompt:**
```
Use the submit_basic_prediction tool with:
- input_file: examples/data/sample.fasta
- output_dir: results/test_claude
- job_name: claude_test_job
```

**Expected Result:**
```json
{
  "status": "submitted",
  "job_id": "<job_id>",
  "message": "Job submitted. Use get_job_status('<job_id>') to check progress."
}
```

---

## Test 9: Check Job Status

**Prompt:**
```
Use the get_job_status tool with the job_id from Test 8
```

**Expected Result:**
```json
{
  "job_id": "<job_id>",
  "job_name": "claude_test_job",
  "status": "pending|running|completed|failed",
  "submitted_at": "<timestamp>",
  "started_at": "<timestamp>",
  "completed_at": "<timestamp>"
}
```

---

## Test 10: Get Job Logs

**Prompt:**
```
Use the get_job_log tool with the job_id from Test 8 and tail 20
```

**Expected Result:**
```json
{
  "status": "success",
  "job_id": "<job_id>",
  "log_lines": [...],
  "total_lines": <number>,
  "tail_lines": 20
}
```

---

## Test 11: Submit Batch Prediction

**Prompt:**
```
Use the submit_batch_prediction tool with:
- input_dir: examples/data/batch_test
- output_dir: results/batch_claude
- file_pattern: "*.fasta"
- job_name: claude_batch_test
```

**Expected Result:**
```json
{
  "status": "submitted",
  "job_id": "<batch_job_id>",
  "message": "Job submitted. Use get_job_status('<batch_job_id>') to check progress."
}
```

---

## Test 12: End-to-End Workflow Test

**Prompt:**
```
Please help me analyze a protein sequence:
1. First validate the file examples/data/sample.fasta
2. Then analyze the sequence composition
3. Finally submit a basic structure prediction job
4. Check the status of the submitted job
```

**Expected Result:**
- All 4 steps should execute successfully
- Should demonstrate the full workflow from validation to submission
- Should show proper job tracking

---

## Test 13: Error Recovery Test

**Prompt:**
```
Try to submit a prediction job with an invalid input file path: /fake/file.fasta
What happens and how should I fix it?
```

**Expected Result:**
- Should show appropriate error message
- Should provide guidance on how to fix the issue

---

## Test 14: Multiple Jobs Test

**Prompt:**
```
Submit 2 different prediction jobs:
1. Basic prediction for examples/data/sample.fasta (job name: test_job_1)
2. Basic prediction for examples/data/simple_test.fasta (job name: test_job_2)

Then list all jobs to see both submissions.
```

**Expected Result:**
- Both jobs should submit successfully
- List_jobs should show both jobs with different job_ids
- Should demonstrate concurrent job management

---

## Success Criteria

For the MCP integration to be considered successful:

- [ ] All 11 tools are discoverable
- [ ] Sync tools (validate, analyze, predict_small) execute quickly (< 30 seconds)
- [ ] Submit tools return job_id and proper status messages
- [ ] Job management tools work (list, status, log)
- [ ] Error handling provides clear, helpful messages
- [ ] End-to-end workflow works from validation to submission
- [ ] Multiple jobs can be managed simultaneously
- [ ] Path resolution works for both relative and absolute paths

## Common Issues and Solutions

**Issue: Tools not found**
- Check: `claude mcp list` - should show chai-lab as connected
- Solution: Re-register with `claude mcp remove chai-lab` then `claude mcp add chai-lab -- ...`

**Issue: Permission errors**
- Check: File permissions on examples/data/ and results/ directories
- Solution: `chmod -R 755 examples/ results/`

**Issue: Import errors in tools**
- Check: Python environment has all dependencies
- Solution: Verify environment with `which python` and check imports

**Issue: Jobs stuck in pending**
- Check: Job manager working with `ls -la jobs/`
- Solution: Check job logs and ensure scripts are executable