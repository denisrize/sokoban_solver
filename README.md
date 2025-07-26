# Sokoban Puzzle Solver - Heuristic Search Algorithm Comparison

## Overview

This project implements and compares multiple heuristic search algorithms to solve Sokoban puzzles optimally. Sokoban is a classic puzzle game where a player pushes boxes in a warehouse to designated goal positions, providing an excellent testbed for analyzing the effectiveness of different admissible heuristics and search strategies in artificial intelligence.

The project includes a comprehensive experimental framework designed for large-scale performance analysis and research, with support for cluster computing environments.

## Features

### Search Algorithms
- **A* Search**: Optimal pathfinding with admissible heuristics
- **AWA* (Anytime Weighted A*)**: Iterative algorithm that finds solutions quickly and improves them over time
- **Best-First Search (Greedy)**: Faster but potentially suboptimal search

### Heuristic Functions
1. **Manhattan Distance**: Sum of minimum distances from each box to closest goal
2. **Hungarian Algorithm (Minimum Cost Matching)**: Optimal assignment of boxes to goals using linear programming
3. **Deadlock Detection**: Advanced pruning technique detecting unsolvable states

### Experimental Framework
- Comprehensive test suite with puzzles of varying complexity levels (3-18)
- Performance metrics collection (runtime, memory, nodes expanded)
- Statistical analysis and visualization
- Cluster computing support with SLURM integration
- Parallel experiment execution
- Detailed result analysis and comparison tools

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install numpy scipy matplotlib pandas tqdm
   ```

3. **Verify installation**:
   ```bash
   cd code
   python main.py --mode demo
   ```

## Usage

The project provides several modes of operation:

### Demo Mode (Default)
Run a simple demonstration with a basic puzzle:
```bash
python main.py --mode demo
```

### Comprehensive Experiments
Run full experiments comparing all algorithm-heuristic combinations:
```bash
python main.py --mode experiment
```

### Analysis Mode
Analyze existing experimental results:
```bash
python main.py --mode analysis
```

### Research Questions Analysis
Get specific answers to research questions:
```bash
python main.py --mode questions
```

### Single Puzzle Execution
Run a specific algorithm on a specific puzzle (useful for cluster computing):
```bash
python run_single_puzzle.py --algorithm AWA* --heuristic Hungarian --deadlock --puzzle level_7 --output results/
```

### Cluster Computing
Submit batch jobs using SLURM:
```bash
sbatch sbatch_search.sh
```

## Project Structure

```
sokoban-solver/
├── code/
│   ├── sokoban_solver.py          # Core solver implementation
│   ├── experiment_framework.py    # Experimental framework and analysis
│   ├── main.py                   # Main execution script
│   ├── run_single_puzzle.py     # Single puzzle runner (for cluster)
│   ├── job_manager.py           # Job management utilities
│   ├── analyze_cluster_results.py # Result aggregation and analysis
│   ├── dynamic_job.sh           # Dynamic SLURM job script
│   ├── CLUSTER_WORKFLOW.md      # Cluster usage documentation
│   └── README.md               # This file
├── results/                    # Experimental results (auto-generated)
│   ├── Astar_Hungarian_with_deadlock/
│   ├── AWAstar_Manhattan_no_deadlock/
│   └── ... (other algorithm-heuristic combinations)
├── sokoban_complexity_analysis.ipynb # Comprehensive analysis notebook
├── all_search_results.csv     # Aggregated experimental results
├── all_awa_iterations.csv     # AWA* iteration data
├── sbatch_search.sh          # SLURM batch script
└── plots/                   # Generated visualization plots
```

## Research Questions Addressed

### 1. Heuristic Effectiveness
**Question**: How does the Hungarian assignment algorithm compare to Manhattan distance in terms of search efficiency?

**Investigation**: The Hungarian algorithm provides a more informed heuristic by solving the optimal assignment problem, typically reducing the search space significantly compared to Manhattan distance.

### 2. Algorithm Trade-offs
**Question**: What are the performance characteristics of different search algorithms across puzzle complexity levels?

**Analysis**: 
- **A***: Optimal solutions with higher memory usage
- **AWA***: Fast initial solutions that improve over time, good memory efficiency
- **Best-First**: Fastest search but potentially suboptimal solutions

### 3. Deadlock Impact
**Question**: How significantly does deadlock detection reduce search space and improve performance?

**Measurement**: Deadlock detection can dramatically prune the search space by identifying unsolvable states early, preventing exploration of impossible branches and significantly reducing nodes expanded.

### 4. Scalability Analysis
**Question**: At what puzzle complexity do different approaches become impractical?

**Testing**: The framework includes puzzles of complexity levels 3-18 to identify scalability limits and performance degradation patterns.

### 5. Anytime Algorithm Benefits
**Question**: How does AWA* perform in finding quick solutions vs. optimal solutions?

**Evaluation**: AWA* provides the benefits of finding usable solutions quickly while continuing to search for optimal solutions, making it suitable for time-constrained environments.

## Test Suite

### Simple Puzzles (Levels 3-7)
- Algorithm verification
- Quick performance comparison
- Basic functionality testing

### Medium Puzzles (Levels 8-12)
- Performance comparison under moderate complexity
- Heuristic effectiveness analysis

### Complex Puzzles (Levels 13-18)
- Scalability analysis
- Memory and time constraint testing
- Algorithm limitations identification

## Performance Metrics

The framework collects comprehensive metrics:

- **Solution Quality**: Number of moves in final solution
- **Search Efficiency**: Number of nodes expanded and generated during search
- **Time Complexity**: Actual runtime for puzzle solving
- **Memory Usage**: Peak memory consumption during search
- **Queue Management**: Maximum queue/frontier size
- **Heuristic Accuracy**: Quality of heuristic estimates
- **Iteration Analysis**: For AWA*, tracking solution improvements over time

## Example Output

```
SOKOBAN SOLVER DEMONSTRATION
==================================================

Puzzle:
#####
# @ #
#$$ #
#. .#
#####

AWA* with Hungarian heuristic (with deadlock detection):
  ✓ Solved in 8 moves
  ✓ Path: D -> L -> D -> R -> R -> U -> L -> L
  ✓ Nodes expanded: 12
  ✓ Runtime: 0.005 seconds

Final state:
#####
# @ #
#** #
#. .#
#####
```

## Advanced Features

### Puzzle Format
The solver accepts standard Sokoban notation:
- `#`: Wall
- `@`: Player
- `$`: Box
- `.`: Goal
- `*`: Box on goal
- `+`: Player on goal
- ` `: Empty space

### Cluster Computing Integration
- SLURM batch job support
- Parallel experiment execution
- Result aggregation from distributed runs
- Dynamic job management

### Experimental Analysis
- Statistical significance testing
- Performance scaling analysis
- Comprehensive visualization
- Result export to CSV/JSON formats

### Custom Puzzles
You can add custom puzzles by modifying the `PuzzleGenerator` class in `experiment_framework.py`.

### Extending Algorithms
New search algorithms can be added to the `SearchAlgorithms` class in `sokoban_solver.py`.

### Custom Heuristics
Additional heuristic functions can be implemented in the `HeuristicFunctions` class.

## Technical Implementation

### State Representation
- Efficient hashable game states
- Set-based position tracking
- Immutable state transitions

### Search Optimization
- Priority queue implementation for A*
- Weighted heuristics for AWA*
- Closed set management for cycle detection
- Memory-efficient state storage

### Heuristic Design
- Admissible heuristics ensuring optimality
- Hungarian algorithm for assignment problems
- Deadlock pattern recognition and pruning

### Experimental Infrastructure
- Modular experiment design
- Comprehensive metrics collection
- Statistical analysis tools
- Visualization framework

## Research Applications

This project is designed for:

1. **Algorithm Performance Research**: Comparing search algorithms across different problem complexities
2. **Heuristic Function Analysis**: Evaluating the effectiveness of different heuristic approaches
3. **Scalability Studies**: Understanding how algorithms perform as problem size increases
4. **Educational Purposes**: Demonstrating AI search concepts with a concrete, visual problem domain

## Future Enhancements

Potential extensions to the project:

1. **Advanced Deadlock Detection**: More sophisticated deadlock patterns and detection algorithms
2. **Machine Learning Heuristics**: Learned heuristic functions using neural networks
3. **Parallel Processing**: Multi-threaded search algorithms
4. **GUI Interface**: Visual puzzle solver and editor
5. **Additional Algorithms**: Bidirectional search, beam search, genetic algorithms
6. **Puzzle Generation**: Automatic puzzle creation with controllable difficulty levels
7. **Real-time Visualization**: Live search progress visualization

## Citation

If you use this code in your research, please cite:
```
Sokoban Heuristic Search Algorithm Comparison Framework
[Your Name/Institution]
[Year]
```