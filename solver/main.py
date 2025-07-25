#!/usr/bin/env python3
"""
Main script for running Sokoban Heuristic Search Algorithm Experiments

This script demonstrates the comprehensive comparison of different search algorithms
and heuristic functions for solving Sokoban puzzles.

Usage:
    python main.py --mode [demo|experiment|analysis]
    
    demo: Run a demonstration with a simple puzzle
    experiment: Run comprehensive experiments
    analysis: Analyze existing results
"""

import argparse
import sys
from sokoban_solver import (
    SokobanPuzzle, SearchAlgorithms, HeuristicFunctions, Position, GameState
)
from experiment_framework import (
    ExperimentRunner, AnalysisToolkit, PuzzleGenerator
)

def demo_mode():
    """Demonstrate the solver with a simple puzzle."""
    print("SOKOBAN SOLVER DEMONSTRATION")
    print("=" * 50)
    
    # Simple puzzle for demonstration
    puzzle_string = """
#####
# @ #
#$$ #
#. .#
#####
"""
    
    print("Puzzle:")
    puzzle = SokobanPuzzle(puzzle_string)
    initial_state = puzzle.get_initial_state()
    puzzle.print_state(initial_state)
    
    print("Testing different algorithm-heuristic combinations:")
    print("-" * 50)
    
    algorithms = {
        # 'A*': SearchAlgorithms().a_star,
        # 'IDA*': SearchAlgorithms().ida_star,
        # 'Best-First': SearchAlgorithms().best_first_search,
        'AWA*': SearchAlgorithms().anytime_weighted_a_star
    }
    
    heuristics = {
        'Manhattan': HeuristicFunctions.manhattan_distance,
        'Hungarian': HeuristicFunctions.minimum_cost_matching,
    }
    
    # Test both with and without deadlock detection
    deadlock_options = [
        (False, "without deadlock detection"),
        (True, "with deadlock detection")
    ]
    
    for alg_name, algorithm in algorithms.items():
        for heur_name, heuristic in heuristics.items():
            for use_deadlock, deadlock_desc in deadlock_options:
                print(f"\n{alg_name} with {heur_name} heuristic ({deadlock_desc}):")
                
                search_instance = SearchAlgorithms()
                algorithm_method = getattr(search_instance, algorithm.__name__)
                
                import time
                start_time = time.time()
                solution = algorithm_method(initial_state, heuristic, use_deadlock_detection=use_deadlock)
                end_time = time.time()
                
                if solution:
                    print(f"  ✓ Solved in {solution.moves} moves")
                    print(f"  ✓ Path: {' -> '.join(solution.path)}")
                    print(f"  ✓ Nodes expanded: {search_instance.nodes_expanded}")
                    print(f"  ✓ Runtime: {end_time - start_time:.3f} seconds")
                    
                    if use_deadlock == False:  # Only show final state once
                        print("\nFinal state:")
                        puzzle.print_state(solution)
                else:
                    print("  ✗ Failed to find solution")

def experiment_mode():
    """Run comprehensive experiments."""
    print("COMPREHENSIVE SOKOBAN EXPERIMENTS")
    print("=" * 50)
    
    runner = ExperimentRunner()
    
    print("Starting comprehensive experiments...")
    print("This will test all combinations of:")
    print("- Algorithms: AWA*, IDA*, Best-First Search, A*")
    print("- Heuristics: Manhattan, Hungarian")
    print("- Puzzle complexities: Simple, Medium, Complex, Corner Cases")
    
    # Run experiments with shorter timeout for demo
    results = runner.run_comprehensive_experiments(timeout_seconds=60)
    
    # Save results
    runner.save_results("sokoban_experiment_results.json")
    
    # Generate analysis
    analyzer = AnalysisToolkit(results, runner.run_dir)
    analyzer.generate_detailed_report("sokoban_analysis_report.txt")
    
    # Try to create plots (may fail if matplotlib backend not available)
    try:
        analyzer.create_performance_plots()
        analyzer.create_all_depth_analysis_plots()
        print("All performance and depth analysis plots generated!")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print(f"\nExperiment completed! Results saved to: {runner.run_dir}")
    print(f"Check '{runner.run_dir}/sokoban_analysis_report.txt' for detailed analysis.")

def analysis_mode():
    """Analyze existing results."""
    print("ANALYZING EXISTING RESULTS")
    print("=" * 50)
    
    try:
        runner = ExperimentRunner()
        runner.load_results("sokoban_experiment_results.json")
        
        analyzer = AnalysisToolkit(runner.results, runner.run_dir)
        summary = analyzer.generate_summary_statistics()
        
        print(f"Loaded {len(runner.results)} experiment results")
        print(f"Overall success rate: {summary['success_rate']:.1%}")
        
        print("\nAlgorithm Performance:")
        for alg, metrics in summary['performance_by_algorithm'].items():
            print(f"  {alg}: {metrics['success_rate']:.1%} success rate, "
                  f"avg {metrics['avg_nodes_expanded']:.0f} nodes expanded")
        
        print("\nHeuristic Performance:")
        for heur, metrics in summary['performance_by_heuristic'].items():
            print(f"  {heur}: {metrics['success_rate']:.1%} success rate, "
                  f"accuracy {metrics['avg_heuristic_accuracy']:.3f}")
        
        # Generate fresh analysis
        analyzer.generate_detailed_report("sokoban_analysis_report.txt")
        
        try:
            analyzer.create_performance_plots()
            analyzer.create_all_depth_analysis_plots()
            print("\nAll plots updated!")
        except Exception as e:
            print(f"Could not generate plots: {e}")
            
    except FileNotFoundError:
        print("No existing results found. Run with --mode experiment first.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

def research_questions_demo():
    """Demonstrate answers to the research questions."""
    print("\nRESEARCH QUESTIONS ANALYSIS")
    print("=" * 50)
    
    try:
        runner = ExperimentRunner()
        runner.load_results("sokoban_experiment_results.json")
        analyzer = AnalysisToolkit(runner.results)
        summary = analyzer.generate_summary_statistics()
        
        print("1. HEURISTIC EFFECTIVENESS:")
        print("   How does Hungarian algorithm compare to Manhattan distance?")
        if 'Hungarian' in summary['performance_by_heuristic'] and 'Manhattan' in summary['performance_by_heuristic']:
            hungarian = summary['performance_by_heuristic']['Hungarian']
            manhattan = summary['performance_by_heuristic']['Manhattan']
            print(f"   - Hungarian: {hungarian['avg_nodes_expanded']:.0f} avg nodes expanded")
            print(f"   - Manhattan: {manhattan['avg_nodes_expanded']:.0f} avg nodes expanded")
            print(f"   - Hungarian reduces search by {((manhattan['avg_nodes_expanded'] - hungarian['avg_nodes_expanded']) / manhattan['avg_nodes_expanded'] * 100):.1f}%")
        
        print("\n2. ALGORITHM TRADE-OFFS:")
        print("   Memory vs. time trade-offs between A* and IDA*:")
        if 'A*' in summary['performance_by_algorithm'] and 'IDA*' in summary['performance_by_algorithm']:
            astar = summary['performance_by_algorithm']['A*']
            ida = summary['performance_by_algorithm']['IDA*']
            print(f"   - A*: {astar['avg_memory']:.2f} MB avg memory, {astar['avg_runtime']:.3f}s avg runtime")
            print(f"   - IDA*: {ida['avg_memory']:.2f} MB avg memory, {ida['avg_runtime']:.3f}s avg runtime")
        
        print("\n3. COMBINED HEURISTIC VALUE:")
        if 'Combined' in summary['performance_by_heuristic']:
            combined = summary['performance_by_heuristic']['Combined']
            print(f"   - Combined heuristic: {combined['avg_nodes_expanded']:.0f} avg nodes expanded")
            print(f"   - Success rate: {combined['success_rate']:.1%}")
        
    except FileNotFoundError:
        print("No results available. Run experiments first.")

def cluster_mode():
    """Launch cluster job submission."""
    print("CLUSTER JOB SUBMISSION MODE")
    print("=" * 50)
    
    try:
        from job_manager import JobManager
        
        print("🚀 Initializing job manager...")
        manager = JobManager()
        
        print("📋 Submitting parallel jobs for all combinations...")
        jobs_submitted, jobs_skipped = manager.submit_all_jobs(dry_run=False)
        
        if jobs_submitted > 0:
            print(f"\n✅ Successfully submitted {jobs_submitted} jobs to the cluster!")
            print(f"⏭️  Skipped {jobs_skipped} jobs (already completed)")
            print(f"\n📊 To check job status, run:")
            print(f"   python job_manager.py --action status")
            print(f"\n📈 To analyze results later, run:")
            print(f"   python analyze_cluster_results.py")
        else:
            print(f"\n⏭️  All {jobs_submitted + jobs_skipped} combinations already completed!")
            print(f"📈 You can analyze existing results with:")
            print(f"   python analyze_cluster_results.py")
            
    except ImportError as e:
        print(f"❌ Error importing job manager: {e}")
        print("Make sure job_manager.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error in cluster mode: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Sokoban Heuristic Search Algorithm Comparison"
    )
    parser.add_argument(
        '--mode', 
        choices=['demo', 'experiment', 'analysis', 'questions', 'cluster'],
        default='demo',
        help='Mode to run the program in'
    )
    
    args = parser.parse_args()
    
    print("SOKOBAN PUZZLE SOLVER - HEURISTIC SEARCH COMPARISON")
    print("=" * 60)
    print("This project compares different search algorithms and heuristics")
    print("for solving Sokoban puzzles optimally.")
    print()
    
    if args.mode == 'demo':
        demo_mode()
    elif args.mode == 'experiment':
        experiment_mode()
    elif args.mode == 'analysis':
        analysis_mode()
    elif args.mode == 'questions':
        research_questions_demo()
    elif args.mode == 'cluster':
        cluster_mode()
    
    print("\n" + "=" * 60)
    print("Program completed successfully!")

if __name__ == "__main__":
    main() 