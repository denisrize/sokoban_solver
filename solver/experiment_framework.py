import time
import tracemalloc
import json
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sokoban_solver import (
    SokobanPuzzle, SearchAlgorithms, HeuristicFunctions, GameState
)

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    puzzle_name: str
    algorithm: str
    heuristic: str
    solved: bool
    solution_length: int
    nodes_expanded: int
    nodes_generated: int
    max_queue_size: int
    runtime_seconds: float
    memory_peak_mb: float
    heuristic_accuracy: float
    solution_path: List[str] = None

class PuzzleGenerator:
    """Generates test puzzles of varying complexity."""
    
    @staticmethod
    def get_level_puzzles() -> Dict[str, str]:
        """Simple puzzles with 2-3 boxes."""
        return {
#             "simple_1_30_steps": """
#  ####\n
#  # ..#\n
# # $ . #\n
# # #$# #\n
# #@$   #\n
#  #   #\n
#   ###\n
# """.strip(),
            "level_3": """
######\n
#...##\n
##$ ##\n
##@ ##\n
# $$ #\n
#   ##\n
#  ###\n
######\n
""".strip(),
            "level_7": """
########\n
##   ###\n
## #$ ##\n
# $    #\n
#. .#  #\n
### @ ##\n
### # ##\n
###   ##\n
########\n
""".strip(),
            "level_9": """
########\n
##   ###\n
##.#$ ##\n
# $    #\n
#. .#$ #\n
### @ ##\n
### # ##\n
###   ##\n
########\n
""".strip(),
    "level_15": """
########\n
#  #####\n
#     ##\n
#     ##\n
### ####\n
# $$$ ##\n
# ...@ #\n
####   #\n
########\n
    """.strip(),
"level_26": """
#########\n
#####  ##\n
## $   ##\n
#  # #$##\n
#.@.   ##\n
## # #  #\n
##      #\n
##  #####\n
#########\n
""".strip(),
"level_42": """
########\n
###  . #\n
#. $ # #\n
#$ .$  #\n
#   #$##\n
##. @ ##\n
########\n
""".strip(),
"level_50": """
#######\n
#.    #####\n
#.    $...#\n
#  $#   $##\n
#   ##  @ #\n
#    $  $ #\n
########  #\n
        ###\n
""".strip(),
"level_60": """
########\n
###   ##\n
#.@$  ##\n
### $.##\n
#.##$ ##\n
# # . ##\n
#$ *$$.#\n
#   .  #\n
########\n
""".strip(),
"level_70": """
##########\n
####     #\n
#### $ $ #\n
#####@ ###\n
###.# ####\n
# $   ####\n
# .    ###\n
#  # $.###\n
# ## .####\n
##########\n
""".strip(),
"level_72": """
##########\n
#.     ###\n
# #. $ ###\n
#  #######\n
#  ##  ###\n
#  # $@###\n
#  #.$ ###\n
#$##  ####\n
#  .   ###\n
##########\n
""".strip(),
        }
    
    @staticmethod
    def get_medium_puzzles() -> Dict[str, str]:
        """Medium puzzles with 4-6 boxes."""
        return {
            "medium_1_45_steps": """
#######\n
#.    #####\n
#.    $...#\n
#  $#   $##\n
#   ##  @ #\n
#    $  $ #\n
########  #\n
        ###\n
""".strip(),
}
    
    @staticmethod
    def get_complex_puzzles() -> Dict[str, str]:
        """Complex 10x10 puzzles with 8+ boxes."""
        return {
            "complex_1_60_steps": """
 ####\n
 #  #####\n
 # $ $  #\n
## #    #\n
#    .###\n
# $#..#\n
##@  ##\n
 #####\n
""".strip(),
        }

    
    @staticmethod
    def get_all_puzzles() -> Dict[str, str]:
        """Get all test puzzles."""
        all_puzzles = {}
        all_puzzles.update(PuzzleGenerator.get_level_puzzles())
        return all_puzzles

class ExperimentRunner:
    """Runs comprehensive experiments comparing algorithms and heuristics."""
    
    def __init__(self):
        self.results: List[ExperimentResult] = []
        self.run_dir: str = None  # Will be set when creating a new run
        self.algorithms = {
            'A*': SearchAlgorithms().a_star,
            'IDA*': SearchAlgorithms().ida_star_optimized,
            'Best-First': SearchAlgorithms().best_first_search,
            'AWA*': SearchAlgorithms().anytime_weighted_a_star
        }
        # Heuristic variants: (name, function, use_deadlock_detection)
        self.heuristics = [
            ('Manhattan', HeuristicFunctions.manhattan_distance, False),
            ('Manhattan+Deadlock', HeuristicFunctions.manhattan_distance, True),
            ('Hungarian', HeuristicFunctions.minimum_cost_matching, False),
            ('Hungarian+Deadlock', HeuristicFunctions.minimum_cost_matching, True),
        ]
    
    def create_run_directory(self) -> str:
        """Create a new run directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        
        # Create runs directory if it doesn't exist
        runs_dir = "runs"
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)
        
        # Create specific run directory
        self.run_dir = os.path.join(runs_dir, run_name)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        print(f"Created run directory: {self.run_dir}")
        return self.run_dir
    
    def run_single_experiment(self, puzzle_name: str, puzzle_string: str, 
                            algorithm_name: str, heuristic_name: str, heuristic_func, use_deadlock_detection: bool,
                            timeout_seconds: float = 300.0) -> List[ExperimentResult]:
        """
        Run a single experiment with given configuration.
        
        Returns a list of ExperimentResult objects:
        - For regular algorithms: single result
        - For AWA*: multiple results (one per iteration)
        """
        
        puzzle = SokobanPuzzle(puzzle_string, puzzle_name)
        initial_state = puzzle.get_initial_state()
        
        algorithm = self.algorithms[algorithm_name]
        
        # Initialize search algorithm instance for statistics
        search_instance = SearchAlgorithms()
        algorithm_method = getattr(search_instance, algorithm.__name__)
        
        # Start memory and time tracking
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Run the search algorithm
            solution = algorithm_method(initial_state, heuristic_func, use_deadlock_detection=use_deadlock_detection)
            end_time = time.time()
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            total_runtime = end_time - start_time
            memory_peak_mb = peak / 1024 / 1024
            
            # Handle AWA* specially - create result for each iteration
            if algorithm_name == 'AWA*' and hasattr(search_instance, 'awa_iterations'):
                results = []
                cumulative_time = 0
                
                for iteration_data in search_instance.awa_iterations:
                    cumulative_time += iteration_data['runtime_seconds']
                    
                    # Calculate heuristic accuracy for this iteration's solution
                    if iteration_data['solved'] and iteration_data['solution_length'] > 0:
                        heuristic_accuracy = heuristic_func(initial_state) / iteration_data['solution_length']
                    else:
                        heuristic_accuracy = 0
                    
                    # Get solution path for this iteration (if it's the best so far)
                    if (iteration_data['solved'] and 
                        search_instance.awa_best_solution and 
                        search_instance.awa_best_solution.moves == iteration_data['solution_length']):
                        solution_path = search_instance.awa_best_solution.path
                    else:
                        solution_path = []
                    
                    result = ExperimentResult(
                        puzzle_name=puzzle_name,
                        algorithm=f"AWA*_iter_{iteration_data['iteration']}_w{iteration_data['weight']:.2f}",
                        heuristic=heuristic_name,
                        solved=iteration_data['solved'],
                        solution_length=iteration_data['solution_length'],
                        nodes_expanded=iteration_data['nodes_expanded'],
                        nodes_generated=iteration_data['nodes_generated'],
                        max_queue_size=search_instance.max_queue_size,  # This is cumulative
                        runtime_seconds=iteration_data['runtime_seconds'],  # Individual iteration time
                        memory_peak_mb=memory_peak_mb,  # Memory is tracked for entire run
                        heuristic_accuracy=heuristic_accuracy,
                        solution_path=solution_path
                    )
                    results.append(result)
                
                # Also add a summary result for the entire AWA* run
                final_result = ExperimentResult(
                    puzzle_name=puzzle_name,
                    algorithm=f"AWA*_total",
                    heuristic=heuristic_name,
                    solved=solution is not None,
                    solution_length=solution.moves if solution else 0,
                    nodes_expanded=search_instance.nodes_expanded,
                    nodes_generated=search_instance.nodes_generated,
                    max_queue_size=search_instance.max_queue_size,
                    runtime_seconds=total_runtime,
                    memory_peak_mb=memory_peak_mb,
                    heuristic_accuracy=heuristic_func(initial_state) / solution.moves if solution and solution.moves > 0 else 0,
                    solution_path=solution.path if solution else []
                )
                results.append(final_result)
                
                return results
            
            # Handle regular algorithms
            else:
                if solution:
                    solution_length = solution.moves
                    heuristic_accuracy = heuristic_func(initial_state) / solution_length if solution_length > 0 else 0
                    solution_path = solution.path
                    solved = True
                else:
                    solution_length = 0
                    heuristic_accuracy = 0
                    solution_path = []
                    solved = False
                
                result = ExperimentResult(
                    puzzle_name=puzzle_name,
                    algorithm=algorithm_name,
                    heuristic=heuristic_name,
                    solved=solved,
                    solution_length=solution_length,
                    nodes_expanded=search_instance.nodes_expanded,
                    nodes_generated=search_instance.nodes_generated,
                    max_queue_size=search_instance.max_queue_size,
                    runtime_seconds=total_runtime,
                    memory_peak_mb=memory_peak_mb,
                    heuristic_accuracy=heuristic_accuracy,
                    solution_path=solution_path
                )
                
                return [result]  # Return as list for consistency
            
        except Exception as e:
            tracemalloc.stop()
            print(f"Error in experiment {puzzle_name}-{algorithm_name}-{heuristic_name}: {e}")
            error_result = ExperimentResult(
                puzzle_name=puzzle_name,
                algorithm=algorithm_name,
                heuristic=heuristic_name,
                solved=False,
                solution_length=0,
                nodes_expanded=0,
                nodes_generated=0,
                max_queue_size=0,
                runtime_seconds=time.time() - start_time,
                memory_peak_mb=0,
                heuristic_accuracy=0,
                solution_path=[]
            )
            return [error_result]
    
    def run_comprehensive_experiments(self, timeout_seconds: float = 300.0) -> List[ExperimentResult]:
        """Run comprehensive experiments across all combinations."""
        
        # Create run directory for this experiment session
        self.create_run_directory()
        
        puzzles = PuzzleGenerator.get_all_puzzles()
        self.results = []
        
        total_experiments = len(puzzles) * len(self.algorithms) * len(self.heuristics)
        
        print(f"Running {total_experiments} experiments...", flush=True)
        
        # Create progress bar
        with tqdm(total=total_experiments, desc="Running experiments", unit="exp") as pbar:
            for puzzle_name, puzzle_string in puzzles.items():
                for algorithm_name in self.algorithms.keys():
                    for heuristic_name, heuristic_func, use_deadlock_detection in self.heuristics:
                        # Update progress bar description
                        pbar.set_description(f"{puzzle_name} - {algorithm_name} - {heuristic_name}")
                        
                        results = self.run_single_experiment(
                            puzzle_name, puzzle_string, algorithm_name, 
                            heuristic_name, heuristic_func, use_deadlock_detection, timeout_seconds
                        )
                        self.results.extend(results)
                        
                        # Update progress bar with result info
                        # For AWA*, show the best result (typically the last one or the total)
                        if algorithm_name == 'AWA*':
                            # Show the AWA*_total result if available, otherwise the best iteration
                            display_result = None
                            for result in results:
                                if result.algorithm == 'AWA*_total':
                                    display_result = result
                                    break
                            if display_result is None and results:
                                # Find best solved result
                                solved_results = [r for r in results if r.solved]
                                if solved_results:
                                    display_result = min(solved_results, key=lambda x: x.solution_length)
                                else:
                                    display_result = results[0]
                            
                            if display_result and display_result.solved:
                                pbar.set_postfix({
                                    'Status': f'✓ AWA* ({len(results)-1} iters)',
                                    'Best': display_result.solution_length,
                                    'Total Nodes': display_result.nodes_expanded,
                                    'Time': f"{display_result.runtime_seconds:.2f}s"
                                })
                            else:
                                pbar.set_postfix({'Status': '✗ AWA* Failed'})
                        else:
                            # Regular algorithm - single result
                            result = results[0]
                            if result.solved:
                                pbar.set_postfix({
                                    'Status': '✓ Solved',
                                    'Moves': result.solution_length,
                                    'Nodes': result.nodes_expanded,
                                    'Time': f"{result.runtime_seconds:.2f}s"
                                })
                            else:
                                pbar.set_postfix({'Status': '✗ Failed'})
                        
                        pbar.update(1)
        
        return self.results
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save experiment results to JSON file."""
        if self.run_dir:
            filepath = os.path.join(self.run_dir, filename)
        else:
            filepath = filename
            
        results_dict = [asdict(result) for result in self.results]
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def load_results(self, filename: str = "experiment_results.json"):
        """Load experiment results from JSON file."""
        # Try to load from run directory first, then current directory
        if self.run_dir and os.path.exists(os.path.join(self.run_dir, filename)):
            filepath = os.path.join(self.run_dir, filename)
        elif os.path.exists(filename):
            filepath = filename
        else:
            raise FileNotFoundError(f"Could not find {filename} in current directory or run directory")
            
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.results = []
        for result_dict in results_dict:
            self.results.append(ExperimentResult(**result_dict))
        
        print(f"Loaded {len(self.results)} results from {filepath}")
    
    def set_run_directory(self, run_dir: str):
        """Set the run directory for saving files."""
        self.run_dir = run_dir

class AnalysisToolkit:
    """Tools for analyzing and visualizing experiment results."""
    
    def __init__(self, results: List[ExperimentResult], run_dir: str = None):
        self.results = results
        self.df = pd.DataFrame([asdict(result) for result in results])
        self.run_dir = run_dir
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        solved_df = self.df[self.df['solved'] == True]
        
        summary = {
            'total_experiments': len(self.results),
            'solved_experiments': len(solved_df),
            'success_rate': len(solved_df) / len(self.results),
            
            'performance_by_algorithm': {},
            'performance_by_heuristic': {},
            'performance_by_puzzle': {},
        }
        
        # Algorithm performance
        for algorithm in self.df['algorithm'].unique():
            alg_df = solved_df[solved_df['algorithm'] == algorithm]
            if len(alg_df) > 0:
                summary['performance_by_algorithm'][algorithm] = {
                    'success_rate': len(alg_df) / len(self.df[self.df['algorithm'] == algorithm]),
                    'avg_solution_length': alg_df['solution_length'].mean(),
                    'avg_nodes_expanded': alg_df['nodes_expanded'].mean(),
                    'avg_runtime': alg_df['runtime_seconds'].mean(),
                    'avg_memory': alg_df['memory_peak_mb'].mean(),
                }
        
        # Heuristic performance
        for heuristic in self.df['heuristic'].unique():
            heur_df = solved_df[solved_df['heuristic'] == heuristic]
            if len(heur_df) > 0:
                summary['performance_by_heuristic'][heuristic] = {
                    'success_rate': len(heur_df) / len(self.df[self.df['heuristic'] == heuristic]),
                    'avg_solution_length': heur_df['solution_length'].mean(),
                    'avg_nodes_expanded': heur_df['nodes_expanded'].mean(),
                    'avg_runtime': heur_df['runtime_seconds'].mean(),
                    'avg_heuristic_accuracy': heur_df['heuristic_accuracy'].mean(),
                }
        
        # Puzzle complexity analysis
        for puzzle in self.df['puzzle_name'].unique():
            puzzle_df = solved_df[solved_df['puzzle_name'] == puzzle]
            if len(puzzle_df) > 0:
                summary['performance_by_puzzle'][puzzle] = {
                    'success_rate': len(puzzle_df) / len(self.df[self.df['puzzle_name'] == puzzle]),
                    'avg_solution_length': puzzle_df['solution_length'].mean(),
                    'avg_nodes_expanded': puzzle_df['nodes_expanded'].mean(),
                    'difficulty_score': puzzle_df['nodes_expanded'].mean() * puzzle_df['runtime_seconds'].mean(),
                }
        
        return summary
    
    def create_performance_plots(self):
        """Create comprehensive performance visualization plots."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        solved_df = self.df[self.df['solved'] == True]
        
        # Create heuristic name mapping for shorter labels
        heuristic_mapping = {
            'Manhattan': 'Man',
            'Manhattan+Deadlock': 'Man+DL',
            'Hungarian': 'CostMin',
            'Hungarian+Deadlock': 'CostMin+DL'
        }
        
        # Apply mapping to dataframes
        solved_df_mapped = solved_df.copy()
        solved_df_mapped['heuristic_short'] = solved_df_mapped['heuristic'].map(heuristic_mapping)
        df_mapped = self.df.copy()
        df_mapped['heuristic_short'] = df_mapped['heuristic'].map(heuristic_mapping)
        
        # 1. Success rate by algorithm
        success_rates = self.df.groupby('algorithm')['solved'].mean()
        axes[0, 0].bar(success_rates.index, success_rates.values)
        axes[0, 0].set_title('Success Rate by Algorithm')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Average nodes expanded by heuristic
        if len(solved_df) > 0:
            avg_nodes = solved_df_mapped.groupby('heuristic_short')['nodes_expanded'].mean()
            axes[0, 1].bar(avg_nodes.index, avg_nodes.values)
            axes[0, 1].set_title('Average Nodes Expanded by Heuristic')
            axes[0, 1].set_ylabel('Nodes Expanded')
            axes[0, 1].tick_params(axis='x', rotation=15)
        
        # 3. Runtime comparison
        if len(solved_df) > 0:
            sns.boxplot(data=solved_df, x='algorithm', y='runtime_seconds', ax=axes[0, 2])
            axes[0, 2].set_title('Runtime Distribution by Algorithm')
            axes[0, 2].set_ylabel('Runtime (seconds)')
        
        # 4. Memory usage comparison
        if len(solved_df) > 0:
            sns.boxplot(data=solved_df_mapped, x='heuristic_short', y='memory_peak_mb', ax=axes[1, 0])
            axes[1, 0].set_title('Memory Usage by Heuristic')
            axes[1, 0].set_ylabel('Peak Memory (MB)')
            axes[1, 0].tick_params(axis='x', rotation=15)
        
        # 5. Solution quality comparison
        if len(solved_df) > 0:
            sns.boxplot(data=solved_df, x='algorithm', y='solution_length', ax=axes[1, 1])
            axes[1, 1].set_title('Solution Length by Algorithm')
            axes[1, 1].set_ylabel('Solution Length (moves)')
        
        # 6. Heuristic accuracy
        if len(solved_df) > 0:
            avg_accuracy = solved_df_mapped.groupby('heuristic_short')['heuristic_accuracy'].mean()
            axes[1, 2].bar(avg_accuracy.index, avg_accuracy.values)
            axes[1, 2].set_title('Average Heuristic Accuracy')
            axes[1, 2].set_ylabel('Accuracy Ratio')
            axes[1, 2].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        
        # Save to run directory if available
        if self.run_dir:
            plot_path = os.path.join(self.run_dir, 'performance_analysis.png')
        else:
            plot_path = 'performance_analysis.png'
            
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance analysis plot saved to {plot_path}")
    
    def generate_detailed_report(self, filename: str = "detailed_analysis_report.txt"):
        """Generate a detailed text report of the analysis."""
        summary = self.generate_summary_statistics()
        
        # Save to run directory if available
        if self.run_dir:
            filepath = os.path.join(self.run_dir, filename)
        else:
            filepath = filename
        
        with open(filepath, 'w') as f:
            f.write("SOKOBAN HEURISTIC SEARCH ALGORITHM COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Experiments: {summary['total_experiments']}\n")
            f.write(f"Solved Experiments: {summary['solved_experiments']}\n")
            f.write(f"Overall Success Rate: {summary['success_rate']:.1%}\n\n")
            
            f.write("ALGORITHM PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            for alg, metrics in summary['performance_by_algorithm'].items():
                f.write(f"{alg}:\n")
                f.write(f"  Success Rate: {metrics['success_rate']:.1%}\n")
                f.write(f"  Avg Solution Length: {metrics['avg_solution_length']:.1f} moves\n")
                f.write(f"  Avg Nodes Expanded: {metrics['avg_nodes_expanded']:.0f}\n")
                f.write(f"  Avg Runtime: {metrics['avg_runtime']:.3f} seconds\n")
                f.write(f"  Avg Memory Usage: {metrics['avg_memory']:.2f} MB\n\n")
            
            f.write("HEURISTIC PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            for heur, metrics in summary['performance_by_heuristic'].items():
                f.write(f"{heur}:\n")
                f.write(f"  Success Rate: {metrics['success_rate']:.1%}\n")
                f.write(f"  Avg Solution Length: {metrics['avg_solution_length']:.1f} moves\n")
                f.write(f"  Avg Nodes Expanded: {metrics['avg_nodes_expanded']:.0f}\n")
                f.write(f"  Avg Runtime: {metrics['avg_runtime']:.3f} seconds\n")
                f.write(f"  Avg Heuristic Accuracy: {metrics['avg_heuristic_accuracy']:.3f}\n\n")
            
            f.write("PUZZLE COMPLEXITY ANALYSIS\n")
            f.write("-" * 35 + "\n")
            puzzles_by_difficulty = sorted(
                summary['performance_by_puzzle'].items(),
                key=lambda x: x[1]['difficulty_score'],
                reverse=True
            )
            for puzzle, metrics in puzzles_by_difficulty:
                f.write(f"{puzzle}:\n")
                f.write(f"  Success Rate: {metrics['success_rate']:.1%}\n")
                f.write(f"  Avg Solution Length: {metrics['avg_solution_length']:.1f} moves\n")
                f.write(f"  Avg Nodes Expanded: {metrics['avg_nodes_expanded']:.0f}\n")
                f.write(f"  Difficulty Score: {metrics['difficulty_score']:.2f}\n\n")
        
        print(f"Detailed report saved to {filepath}")
    
    def analyze_depth_vs_expansion(self):
        """Analyze relationship between solution depth and nodes expanded."""
        solved_df = self.df[self.df['solved'] == True]
        
        if len(solved_df) == 0:
            print("No solved puzzles to analyze.")
            return
        
        plt.figure(figsize=(12, 8))
        
        for algorithm in solved_df['algorithm'].unique():
            alg_data = solved_df[solved_df['algorithm'] == algorithm]
            plt.scatter(alg_data['solution_length'], alg_data['nodes_expanded'], 
                       label=algorithm, alpha=0.7, s=60)
        
        plt.xlabel('Solution Depth (moves)', fontsize=12)
        plt.ylabel('Nodes Expanded', fontsize=12)
        plt.title('Search Efficiency: Solution Depth vs Nodes Expanded', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.yscale('log')  # Log scale due to exponential growth
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to run directory if available
        if self.run_dir:
            plot_path = os.path.join(self.run_dir, 'depth_vs_nodes_expanded.png')
        else:
            plot_path = 'depth_vs_nodes_expanded.png'
            
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Depth vs Nodes Expanded plot saved as '{plot_path}'")
    
    def analyze_depth_vs_memory(self):
        """Analyze relationship between solution depth and memory usage."""
        solved_df = self.df[self.df['solved'] == True]
        
        if len(solved_df) == 0:
            print("No solved puzzles to analyze.")
            return
        
        plt.figure(figsize=(12, 8))
        
        for algorithm in solved_df['algorithm'].unique():
            alg_data = solved_df[solved_df['algorithm'] == algorithm]
            plt.scatter(alg_data['solution_length'], alg_data['memory_peak_mb'], 
                       label=algorithm, alpha=0.7, s=60)
        
        plt.xlabel('Solution Depth (moves)', fontsize=12)
        plt.ylabel('Peak Memory Usage (MB)', fontsize=12)
        plt.title('Memory Efficiency: Solution Depth vs Peak Memory Usage', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to run directory if available
        if self.run_dir:
            plot_path = os.path.join(self.run_dir, 'depth_vs_memory_usage.png')
        else:
            plot_path = 'depth_vs_memory_usage.png'
            
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Depth vs Memory Usage plot saved as '{plot_path}'")
    
    def analyze_depth_vs_runtime(self):
        """Analyze relationship between solution depth and runtime."""
        solved_df = self.df[self.df['solved'] == True]
        
        if len(solved_df) == 0:
            print("No solved puzzles to analyze.")
            return
        
        plt.figure(figsize=(12, 8))
        
        for algorithm in solved_df['algorithm'].unique():
            alg_data = solved_df[solved_df['algorithm'] == algorithm]
            plt.scatter(alg_data['solution_length'], alg_data['runtime_seconds'], 
                       label=algorithm, alpha=0.7, s=60)
        
        plt.xlabel('Solution Depth (moves)', fontsize=12)
        plt.ylabel('Runtime (seconds)', fontsize=12)
        plt.title('Time Efficiency: Solution Depth vs Runtime', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.yscale('log')  # Log scale for better visualization of runtime differences
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to run directory if available
        if self.run_dir:
            plot_path = os.path.join(self.run_dir, 'depth_vs_runtime.png')
        else:
            plot_path = 'depth_vs_runtime.png'
            
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Depth vs Runtime plot saved as '{plot_path}'")
    
    def create_all_depth_analysis_plots(self):
        """Create all three depth analysis plots."""
        print("Creating depth analysis plots...")
        self.analyze_depth_vs_expansion()
        self.analyze_depth_vs_memory()
        self.analyze_depth_vs_runtime()
        print("All depth analysis plots completed!")

if __name__ == "__main__":
    # Example usage
    runner = ExperimentRunner()
    results = runner.run_comprehensive_experiments(timeout_seconds=60)
    runner.save_results("sokoban_experiment_results.json")
    
    analyzer = AnalysisToolkit(results, runner.run_dir)
    analyzer.create_performance_plots()
    analyzer.create_all_depth_analysis_plots()
    analyzer.generate_detailed_report("sokoban_analysis_report.txt")
    
    print(f"\nAll files saved to: {runner.run_dir}") 