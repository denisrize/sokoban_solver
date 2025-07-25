# Sokoban Solver: Parallel Cluster Workflow

This document describes how to run the Sokoban experiments in parallel on a SLURM cluster.

## Overview

The parallel workflow submits separate jobs for each **puzzle-combination pair**:
- **Algorithms**: A*, IDA*, Best-First Search
- **Heuristics**: Manhattan Distance, Hungarian Algorithm  
- **Deadlock Detection**: ON/OFF
- **Puzzles**: All test puzzles (simple, medium, complex, corner cases)

This creates **individual jobs** for each puzzle with each combination (12 combinations × number of puzzles), allowing maximum parallelization and the ability to skip already-solved puzzles.

## File Structure

```
search_project/
├── code/                      # All source code
│   ├── job_manager.py         # Submits parallel jobs
│   ├── run_single_puzzle.py   # Runs single puzzle
│   ├── analyze_cluster_results.py # Analyzes all results
│   ├── main.py                # Updated with cluster mode
│   ├── sokoban_solver.py      # Core solver
│   └── experiment_framework.py # Experiment utilities
├── jobs_output/               # All job outputs and results
│   └── batch_YYYYMMDD_HHMMSS/ # Timestamped batch
│       ├── A_star_Manhattan_no_deadlock/
│       │   ├── simple_1_result.json
│       │   ├── medium_1_result.json
│       │   ├── complex_1_result.json
│       │   ├── deadlock_corner_result.json
│       │   ├── puzzle_summary.txt
│       │   └── simple_1_job_12345.out
│       ├── A_star_Manhattan_with_deadlock/
│       └── ... (12 total combination directories)
├── sbatch_search.sh           # Original batch script
├── organize_project.py        # Reorganization script
└── CLUSTER_WORKFLOW.md        # This documentation
```

## Workflow Steps

### 1. Reorganize Project (First Time Setup)

```bash
# Reorganize existing project structure (run once)
python organize_project.py
```

### 2. Submit Jobs to Cluster

```bash
# Submit all jobs (skips already completed combinations)
python code/main.py --mode cluster

# Or use job manager directly
python code/job_manager.py --action submit

# Dry run to see what would be submitted
python code/job_manager.py --action dry-run
```

This will:
- ✅ Create timestamped batch directory
- ✅ Generate sbatch scripts for each combination
- ✅ Submit jobs to SLURM scheduler
- ✅ Skip combinations that already have results
- ✅ Track submitted job IDs

### 3. Monitor Job Status

```bash
# Check job status
python code/job_manager.py --action status

# Or use SLURM commands directly
squeue -u $USER
```

### 4. Analyze Results (After Jobs Complete)

```bash
# Full analysis with plots and reports
python code/analyze_cluster_results.py --action analyze

# Quick summary only
python code/analyze_cluster_results.py --action summary

# Generate plots only
python code/analyze_cluster_results.py --action plots
```

## Results Structure

Each combination creates its own subdirectory with:

- **`<puzzle_name>_result.json`**: Individual puzzle result (one file per puzzle)
- **`puzzle_summary.txt`**: Log of all puzzle results for this combination
- **`<puzzle_name>_job_*.out`**: SLURM output logs (one per puzzle job)
- **`<puzzle_name>_job_*.err`**: SLURM error logs (one per puzzle job)

## Job Configuration

Individual jobs are configured with:
- **Time limit**: 4 hours (per puzzle)
- **Memory**: 8GB (reduced since only one puzzle per job)
- **CPUs**: 1
- **Timeout per puzzle**: 1 hour
- **Partition**: main

## Key Features

### ✅ Smart Job Management
- Only submits jobs for missing puzzle-combination pairs
- Prevents duplicate work if re-run  
- Individual puzzle-level granularity
- Timestamped batches for organization

### ✅ Massive Parallel Execution
- Each puzzle runs independently with each combination
- Maximum cluster utilization (hundreds of small jobs vs 12 large jobs)
- Individual failure isolation (one puzzle failure doesn't affect others)
- Efficient resource usage with smaller memory footprint per job

### ✅ Comprehensive Analysis
- Aggregates results from all jobs
- Creates comparison plots and heatmaps
- Generates detailed performance reports

### ✅ Deadlock Detection Comparison
- Tests each algorithm/heuristic with and without deadlock detection
- Quantifies pruning effectiveness
- Compares memory and time benefits

## Example Usage

```bash
# 1. Reorganize project (first time only)
python organize_project.py

# 2. Submit all jobs
python code/main.py --mode cluster

# 3. Wait for jobs to complete (check with squeue)
squeue -u $USER

# 4. Analyze results
python code/analyze_cluster_results.py

# 5. View comprehensive report
cat jobs_output/batch_*/comprehensive_analysis.txt

# 6. View plots
ls jobs_output/batch_*/analysis_plots/
```

## Output Files

After analysis, you'll have:

- **`comprehensive_analysis.txt`**: Detailed performance comparison
- **`analysis_plots/performance_comparison.png`**: Algorithm/heuristic performance
- **`analysis_plots/deadlock_comparison.png`**: Deadlock detection impact  
- **`analysis_plots/algorithm_heuristic_heatmap.png`**: Success rate heatmaps
- **`analysis_plots/depth_analysis.png`**: Solution depth analysis

## Troubleshooting

### Job Submission Issues
```bash
# Check if sbatch is available
which sbatch

# Verify job manager works
python code/job_manager.py --action dry-run
```

### Missing Results
```bash
# Check job logs
cat jobs_output/batch_*/*/job_*.out
cat jobs_output/batch_*/*/job_*.err

# Re-submit failed jobs
python code/job_manager.py --action submit
```

### Analysis Issues
```bash
# Check if results exist
find jobs_output -name "*_result.json"

# Manual analysis
python code/analyze_cluster_results.py --action summary
```

## Research Questions Addressed

This workflow enables answering:

1. **Which algorithm is most efficient?** (A* vs IDA* vs Best-First)
2. **How much do advanced heuristics help?** (Manhattan vs Hungarian)
3. **What's the impact of deadlock detection?** (ON vs OFF comparison)
4. **Which combinations work best for different puzzle types?**
5. **What are the memory vs time trade-offs?**

The parallel execution allows comprehensive evaluation across all combinations simultaneously, providing robust statistical comparisons for your research. 