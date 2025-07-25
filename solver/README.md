# Sokoban Puzzle Solver - Heuristic Search Algorithm Comparison

## Overview

This project implements and compares multiple heuristic search algorithms to solve Sokoban puzzles optimally. Sokoban is a classic puzzle game where a player pushes boxes in a warehouse to designated goal positions, providing an excellent testbed for analyzing the effectiveness of different admissible heuristics and search strategies in artificial intelligence.

## Features

### Search Algorithms
- **A* Search**: Optimal pathfinding with admissible heuristics
- **IDA* (Iterative Deepening A*)**: Memory-efficient alternative to A*
- **Best-First Search (Greedy)**: Faster but potentially suboptimal search

### Heuristic Functions
1. **Manhattan Distance**: Sum of minimum distances from each box to closest goal
2. **Minimum Cost Matching (Hungarian Algorithm)**: Optimal assignment of boxes to goals
3. **Deadlock Detection**: Advanced heuristic detecting unsolvable states
4. **Combined Heuristic**: Maximum of multiple admissible heuristics

### Experimental Framework
- Comprehensive test suite with puzzles of varying complexity
- Performance metrics collection (runtime, memory, nodes expanded)
- Statistical analysis and visualization
- Research question investigation

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
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

## Project Structure

```
sokoban-solver/
├── sokoban_solver.py          # Core solver implementation
├── experiment_framework.py    # Experimental framework and analysis
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── results/                 # Generated results (created after experiments)
    ├── sokoban_experiment_results.json
    ├── sokoban_analysis_report.txt
    └── performance_analysis.png
```

## Research Questions Addressed

### 1. Heuristic Effectiveness
**Question**: How does the assignment problem heuristic compare to simpler Manhattan distance in terms of nodes expanded?

**Investigation**: The Hungarian algorithm provides a more informed heuristic by solving the optimal assignment problem, typically reducing the search space significantly compared to Manhattan distance.

### 2. Algorithm Trade-offs
**Question**: What are the memory vs. time trade-offs between A* and IDA*?

**Analysis**: 
- **A***: Higher memory usage but potentially faster due to avoiding repeated work
- **IDA***: Lower memory usage but may repeat computations across iterations

### 3. Deadlock Impact
**Question**: How significantly does deadlock detection reduce search space?

**Measurement**: Deadlock detection can dramatically prune the search space by identifying unsolvable states early, preventing exploration of impossible branches.

### 4. Scalability Analysis
**Question**: At what puzzle size do different approaches become impractical?

**Testing**: The framework includes puzzles of increasing complexity to identify scalability limits.

### 5. Combined Heuristic Value
**Question**: Does combining multiple heuristics provide meaningful improvements over individual ones?

**Evaluation**: The combined heuristic takes the maximum of admissible heuristics, providing the most informed lower bound while maintaining optimality.

## Test Suite

### Simple Puzzles (5×5 grids, 2-3 boxes)
- Algorithm verification
- Quick performance comparison
- Basic functionality testing

### Medium Puzzles (7×7 grids, 4-6 boxes)
- Performance comparison under moderate complexity
- Heuristic effectiveness analysis

### Complex Puzzles (10×10 grids, 8+ boxes)
- Scalability analysis
- Memory and time constraint testing

### Corner Cases
- Deadlock scenarios
- Tight corridors
- Multiple solution paths

## Performance Metrics

The framework collects comprehensive metrics:

- **Solution Quality**: Number of moves in final solution
- **Search Efficiency**: Number of nodes expanded during search
- **Time Complexity**: Actual runtime for puzzle solving
- **Memory Usage**: Peak memory consumption during search
- **Heuristic Accuracy**: Ratio of heuristic estimate to actual solution cost

## Example Output

```
SOKOBAN SOLVER DEMONSTRATION
==================================================

Puzzle:
#####
#.@ #
#$$ #
#. .#
#####

A* with Manhattan heuristic:
  ✓ Solved in 8 moves
  ✓ Path: D -> L -> D -> R -> R -> U -> L -> L
  ✓ Nodes expanded: 23
  ✓ Runtime: 0.003 seconds

A* with Hungarian heuristic:
  ✓ Solved in 8 moves
  ✓ Path: D -> L -> D -> R -> R -> U -> L -> L
  ✓ Nodes expanded: 12
  ✓ Runtime: 0.005 seconds
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
- Closed set management
- Memory-efficient IDA* implementation

### Heuristic Design
- Admissible heuristics ensuring optimality
- Hungarian algorithm for assignment problems
- Deadlock pattern recognition

## Future Enhancements

Potential extensions to the project:

1. **Advanced Deadlock Detection**: More sophisticated deadlock patterns
2. **Machine Learning Heuristics**: Learned heuristic functions
3. **Parallel Processing**: Multi-threaded search algorithms
4. **GUI Interface**: Visual puzzle solver and editor
5. **Additional Algorithms**: Bidirectional search, beam search
6. **Puzzle Generation**: Automatic puzzle creation with difficulty levels