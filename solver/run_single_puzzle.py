#!/usr/bin/env python3
"""
Single Puzzle Runner for Sokoban Experiments

This script runs a single puzzle with a single combination of algorithm, heuristic, 
and deadlock detection and saves the result to a specified directory.
"""

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import List

from sokoban_solver import (
    SokobanPuzzle, SearchAlgorithms, HeuristicFunctions, GameState
)
from experiment_framework import ExperimentResult, PuzzleGenerator


class SinglePuzzleRunner:
    """Runs a single puzzle with a single algorithm-heuristic-deadlock combination."""
    
    def __init__(self, algorithm_name: str, heuristic_name: str, use_deadlock_detection: bool, output_dir: str):
        self.algorithm_name = algorithm_name
        self.heuristic_name = heuristic_name
        self.use_deadlock_detection = use_deadlock_detection
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize algorithm and heuristic
        self.search_algorithms = SearchAlgorithms()
        self.algorithm_func = self._get_algorithm_function()
        self.heuristic_func = self._get_heuristic_function()
        
        print(f"Initialized runner for: {algorithm_name} + {heuristic_name} + Deadlock={use_deadlock_detection}")
    
    def _get_algorithm_function(self):
        """Get the algorithm function based on name."""
        algorithm_map = {
            'A*': self.search_algorithms.a_star,
            'IDA*': self.search_algorithms.ida_star_optimized,
            'Best-First': self.search_algorithms.best_first_search,
            'AWA*': self.search_algorithms.anytime_weighted_a_star
        }
        
        if self.algorithm_name not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
        
        return algorithm_map[self.algorithm_name]
    
    def _get_heuristic_function(self):
        """Get the heuristic function based on name."""
        heuristic_map = {
            'Manhattan': HeuristicFunctions.manhattan_distance,
            'Hungarian': HeuristicFunctions.minimum_cost_matching
        }
        
        if self.heuristic_name not in heuristic_map:
            raise ValueError(f"Unknown heuristic: {self.heuristic_name}")
        
        return heuristic_map[self.heuristic_name]
    
    def run_puzzle(self, puzzle_name: str, timeout_seconds: float = 3600.0) -> List[ExperimentResult]:
        """Run experiment on the specified puzzle."""
        
        # Get puzzle string from puzzle generator
        puzzles = PuzzleGenerator.get_all_puzzles()
        if puzzle_name not in puzzles:
            raise ValueError(f"Unknown puzzle: {puzzle_name}")
        
        puzzle_string = puzzles[puzzle_name]
        puzzle = SokobanPuzzle(puzzle_string, puzzle_name)
        initial_state = puzzle.get_initial_state()
        
        print(f"Running puzzle '{puzzle_name}' with {self.algorithm_name} + {self.heuristic_name} + Deadlock={self.use_deadlock_detection}")
        print(f"Puzzle size: {puzzle.width}x{puzzle.height}")
        print(f"Boxes: {len(initial_state.boxes)}, Goals: {len(initial_state.goals)}")
        
        # Reset search statistics
        self.search_algorithms.reset_stats()
        
        # Start memory and time tracking
        if self.algorithm_name != 'AWA*':
            tracemalloc.start()
        start_time = time.time()
        
        try:
            # Run the search algorithm
            solution = self.algorithm_func(
                initial_state, 
                self.heuristic_func, 
                use_deadlock_detection=self.use_deadlock_detection
            )
            end_time = time.time()
            
            # Get memory statistics
            if self.algorithm_name != 'AWA*':
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_peak_mb = peak / 1024 / 1024
            else:
                memory_peak_mb = self.search_algorithms.awa_total_peak_memory_mb
            
            # Calculate metrics
            runtime = end_time - start_time
            
            # Handle AWA* specially - create results for each iteration
            if self.algorithm_name == 'AWA*' and hasattr(self.search_algorithms, 'awa_iterations'):
                results = []
                
                for iteration_data in self.search_algorithms.awa_iterations:
                    # Calculate heuristic accuracy for this iteration's solution
                    if iteration_data['solved'] and iteration_data['solution_length'] > 0:
                        heuristic_accuracy = self.heuristic_func(initial_state) / iteration_data['solution_length']
                    else:
                        heuristic_accuracy = 0
                    
                    # Get solution path for this iteration (if it's the best so far)
                    if (iteration_data['solved'] and 
                        self.search_algorithms.awa_best_solution and 
                        self.search_algorithms.awa_best_solution.moves == iteration_data['solution_length']):
                        solution_path = self.search_algorithms.awa_best_solution.path
                    else:
                        solution_path = []
                    
                    result = ExperimentResult(
                        puzzle_name=puzzle_name,
                        algorithm=f"AWA*_iter_{iteration_data['iteration']}_w{iteration_data['weight']:.2f}",
                        heuristic=self.heuristic_name,
                        solved=iteration_data['solved'],
                        solution_length=iteration_data['solution_length'],
                        nodes_expanded=iteration_data['nodes_expanded'],
                        nodes_generated=iteration_data['nodes_generated'],
                        max_queue_size=self.search_algorithms.max_queue_size,
                        runtime_seconds=iteration_data['runtime_seconds'],
                        memory_peak_mb=iteration_data['memory_peak_mb'],
                        heuristic_accuracy=heuristic_accuracy,
                        solution_path=solution_path
                    )
                    results.append(result)
                    
                    # Print iteration details
                    if iteration_data['solved']:
                        print(f"✅ Iteration {iteration_data['iteration']} (w={iteration_data['weight']:.2f}): SOLVED in {iteration_data['solution_length']} moves!")
                        print(f"   Nodes expanded: {iteration_data['nodes_expanded']:,}")
                        print(f"   Time: {iteration_data['runtime_seconds']:.2f}s")
                    else:
                        print(f"❌ Iteration {iteration_data['iteration']} (w={iteration_data['weight']:.2f}): FAILED")
                        print(f"   Nodes expanded: {iteration_data['nodes_expanded']:,}")
                        print(f"   Time: {iteration_data['runtime_seconds']:.2f}s")
                
                # Add total summary result
                final_result = ExperimentResult(
                    puzzle_name=puzzle_name,
                    algorithm=f"AWA*_total",
                    heuristic=self.heuristic_name,
                    solved=solution is not None,
                    solution_length=solution.moves if solution else 0,
                    nodes_expanded=self.search_algorithms.nodes_expanded,
                    nodes_generated=self.search_algorithms.nodes_generated,
                    max_queue_size=self.search_algorithms.max_queue_size,
                    runtime_seconds=runtime,
                    memory_peak_mb=memory_peak_mb,
                    heuristic_accuracy=self.heuristic_func(initial_state) / solution.moves if solution and solution.moves > 0 else 0,
                    solution_path=solution.path if solution else []
                )
                results.append(final_result)
                
                if solution:
                    print(f"\n🎯 AWA* FINAL RESULT: SOLVED in {solution.moves} moves!")
                    print(f"   Total nodes expanded: {self.search_algorithms.nodes_expanded:,}")
                    print(f"   Total runtime: {runtime:.2f}s")
                    print(f"   Total memory: {memory_peak_mb:.2f}MB")
                    print(f"   Iterations completed: {len(self.search_algorithms.awa_iterations)}")
                else:
                    print(f"\n❌ AWA* FINAL RESULT: FAILED to solve")
                    print(f"   Total nodes expanded: {self.search_algorithms.nodes_expanded:,}")
                    print(f"   Total runtime: {runtime:.2f}s")
                    print(f"   Total memory: {memory_peak_mb:.2f}MB")
                
                return results
            
            # Handle regular algorithms
            else:
                if solution:
                    solution_length = solution.moves
                    heuristic_accuracy = self.heuristic_func(initial_state) / solution_length if solution_length > 0 else 0
                    solution_path = solution.path
                    solved = True
                    print(f"✅ SOLVED in {solution_length} moves!")
                    print(f"   Nodes expanded: {self.search_algorithms.nodes_expanded:,}")
                    print(f"   Runtime: {runtime:.2f}s")
                    print(f"   Memory: {memory_peak_mb:.2f}MB")
                else:
                    solution_length = 0
                    heuristic_accuracy = 0
                    solution_path = []
                    solved = False
                    print(f"❌ FAILED to solve")
                    print(f"   Nodes expanded: {self.search_algorithms.nodes_expanded:,}")
                    print(f"   Runtime: {runtime:.2f}s")
                    print(f"   Memory: {memory_peak_mb:.2f}MB")
                
                result = ExperimentResult(
                    puzzle_name=puzzle_name,
                    algorithm=self.algorithm_name,
                    heuristic=self.heuristic_name,
                    solved=solved,
                    solution_length=solution_length,
                    nodes_expanded=self.search_algorithms.nodes_expanded,
                    nodes_generated=self.search_algorithms.nodes_generated,
                    max_queue_size=self.search_algorithms.max_queue_size,
                    runtime_seconds=runtime,
                    memory_peak_mb=memory_peak_mb,
                    heuristic_accuracy=heuristic_accuracy,
                    solution_path=solution_path
                )
                
                return [result]
            
        except Exception as e:
            print(f"💥 Error: {e}")
            tracemalloc.stop()
            end_time = time.time()
            
            error_result = ExperimentResult(
                puzzle_name=puzzle_name,
                algorithm=self.algorithm_name,
                heuristic=self.heuristic_name,
                solved=False,
                solution_length=0,
                nodes_expanded=0,
                nodes_generated=0,
                max_queue_size=0,
                runtime_seconds=end_time - start_time,
                memory_peak_mb=0,
                heuristic_accuracy=0,
                solution_path=[]
            )
            return [error_result]
    
    def save_results(self, results: List[ExperimentResult]):
        """Save the results to JSON files."""
        
        for result in results:
            # Save individual result
            result_file = self.output_dir / f"{result.puzzle_name}_{result.algorithm}_result.json"
            result_dict = asdict(result)
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
        
        # Create a summary log entry
        summary_file = self.output_dir / "puzzle_summary.txt"
        
        for result in results:
            status = "SOLVED" if result.solved else "FAILED"
            
            with open(summary_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ")
                f.write(f"{result.puzzle_name:15} | {result.algorithm:20} | {status:6} | ")
                if result.solved:
                    f.write(f"Moves: {result.solution_length:3d} | ")
                    f.write(f"Nodes: {result.nodes_expanded:8,} | ")
                    f.write(f"Time: {result.runtime_seconds:6.2f}s | ")
                    f.write(f"Memory: {result.memory_peak_mb:6.2f}MB")
                else:
                    f.write(f"Nodes: {result.nodes_expanded:8,} | ")
                    f.write(f"Time: {result.runtime_seconds:6.2f}s")
                f.write("\n")
        
        print(f"💾 Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Sokoban experiment for single puzzle")
    parser.add_argument('--algorithm', required=True, choices=['A*', 'IDA*', 'Best-First', 'AWA*'],
                       help='Search algorithm to use')
    parser.add_argument('--heuristic', required=True, choices=['Manhattan', 'Hungarian'],
                       help='Heuristic function to use')
    parser.add_argument('--deadlock-detection', type=str, choices=['true', 'false'], required=True,
                       help='Whether to use deadlock detection')
    parser.add_argument('--puzzle-name', required=True,
                       help='Name of the puzzle to solve')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save results')
    parser.add_argument('--timeout', type=float, default=3600.0,
                       help='Timeout in seconds (default: 3600)')
    
    args = parser.parse_args()
    
    # Convert deadlock detection string to boolean
    use_deadlock = args.deadlock_detection.lower() == 'true'
    
    # Create runner and execute
    runner = SinglePuzzleRunner(
        algorithm_name=args.algorithm,
        heuristic_name=args.heuristic,
        use_deadlock_detection=use_deadlock,
        output_dir=args.output_dir
    )
    
    try:
        # Run the puzzle
        results = runner.run_puzzle(args.puzzle_name, timeout_seconds=args.timeout)
        
        # Save results
        runner.save_results(results)
        
        print(f"\n🎉 Puzzle experiment completed!")
        print(f"📂 Results saved to: {args.output_dir}")
        
        # Exit with appropriate code
        exit(0 if all(result.solved for result in results) else 1)
        
    except Exception as e:
        print(f"💥 Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        exit(2)


if __name__ == "__main__":
    main() 