#!/usr/bin/env python3
"""
Cluster Results Analysis Tool

This script analyzes and visualizes results from parallel Sokoban experiments
run on the cluster. It aggregates results from all subdirectories and creates
comprehensive analysis reports and plots.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

from experiment_framework import ExperimentResult, AnalysisToolkit


class ClusterResultsAnalyzer:
    """Analyzes results from parallel cluster experiments."""
    
    def __init__(self, results_base_dir: str = "results"):
        self.results_base_dir = Path(results_base_dir)
        self.all_results: List[ExperimentResult] = []
        self.batch_results: Dict[str, List[ExperimentResult]] = {}
        
    def discover_result_files(self) -> List[Path]:
        """Discover all individual puzzle result files."""
        result_files = []
        
        for combo_dir in self.results_base_dir.glob("*"):
            if combo_dir.is_dir():
                # Find all *_result.json files
                for result_file in combo_dir.glob("*_result.json"):
                    result_files.append(result_file)
        
        return result_files
    
    def load_all_results(self) -> int:
        """Load results from all discovered individual puzzle result files."""
        result_files = self.discover_result_files()
        
        print(f"Found {len(result_files)} individual puzzle result files")
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Convert to ExperimentResult object
                result = ExperimentResult(**result_data)
                
                # Extract batch name and combination name
                batch_name = result_file.parent.parent.name
                combo_name = result_file.parent.name
                
                if batch_name not in self.batch_results:
                    self.batch_results[batch_name] = []
                
                self.batch_results[batch_name].append(result)
                self.all_results.append(result)
                
                status = "✅" if result.solved else "❌"
                print(f"  {status} Loaded {result.puzzle_name} from {batch_name}/{combo_name}")
                
            except Exception as e:
                print(f"  ❌ Error loading {result_file}: {e}")
        
        print(f"Total results loaded: {len(self.all_results)}")
        return len(self.all_results)
    
    def generate_comprehensive_report(self, output_file: str = "comprehensive_analysis.txt"):
        """Generate a comprehensive analysis report."""
        
        if not self.all_results:
            print("No results loaded. Run load_all_results() first.")
            return
        
        # Create analysis toolkit
        analyzer = AnalysisToolkit(self.all_results)
        summary = analyzer.generate_summary_statistics()
        
        output_path = self.results_base_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("COMPREHENSIVE SOKOBAN EXPERIMENT ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"  Total experiments: {summary['total_experiments']}\n")
            f.write(f"  Solved experiments: {summary['solved_experiments']}\n")
            f.write(f"  Success rate: {summary['success_rate']:.1%}\n\n")
            
            # Algorithm performance
            f.write("ALGORITHM PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for alg, metrics in summary['performance_by_algorithm'].items():
                f.write(f"{alg:12} Success: {metrics['success_rate']:.1%}  ")
                f.write(f"Avg Nodes: {metrics['avg_nodes_expanded']:8.0f}  ")
                f.write(f"Avg Time: {metrics['avg_runtime']:6.2f}s  ")
                f.write(f"Avg Memory: {metrics['avg_memory']:6.2f}MB\n")
            
            # Heuristic performance
            f.write("\nHEURISTIC PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for heur, metrics in summary['performance_by_heuristic'].items():
                f.write(f"{heur:20} Success: {metrics['success_rate']:.1%}  ")
                f.write(f"Avg Nodes: {metrics['avg_nodes_expanded']:8.0f}  ")
                f.write(f"Accuracy: {metrics['avg_heuristic_accuracy']:.3f}\n")
            
            # Puzzle difficulty
            f.write("\nPUZZLE DIFFICULTY ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            for puzzle, metrics in summary['performance_by_puzzle'].items():
                f.write(f"{puzzle:20} Success: {metrics['success_rate']:.1%}  ")
                f.write(f"Avg Nodes: {metrics['avg_nodes_expanded']:8.0f}  ")
                f.write(f"Difficulty: {metrics['difficulty_score']:10.1f}\n")
            
            # Deadlock detection analysis
            f.write("\nDEADLOCK DETECTION ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            df = pd.DataFrame([asdict(result) for result in self.all_results])
            solved_df = df[df['solved'] == True]
            
            if len(solved_df) > 0:
                # Group by deadlock detection (inferred from heuristic name)
                deadlock_groups = {}
                for _, row in solved_df.iterrows():
                    has_deadlock = 'Deadlock' in row['heuristic']
                    key = 'With Deadlock' if has_deadlock else 'Without Deadlock'
                    if key not in deadlock_groups:
                        deadlock_groups[key] = []
                    deadlock_groups[key].append(row)
                
                for group_name, group_data in deadlock_groups.items():
                    group_df = pd.DataFrame(group_data)
                    f.write(f"{group_name:20} Count: {len(group_df):4d}  ")
                    f.write(f"Avg Nodes: {group_df['nodes_expanded'].mean():8.0f}  ")
                    f.write(f"Avg Time: {group_df['runtime_seconds'].mean():6.2f}s\n")
        
        print(f"📄 Comprehensive report saved to: {output_path}")
    
    def create_comparison_plots(self, output_dir: str = "analysis_plots"):
        """Create comparison plots for all results."""
        
        if not self.all_results:
            print("No results loaded. Run load_all_results() first.")
            return
        
        plots_dir = self.results_base_dir / output_dir
        plots_dir.mkdir(exist_ok=True)
        
        # Create analysis toolkit and generate plots
        analyzer = AnalysisToolkit(self.all_results)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        
        # Create comprehensive plots
        try:
            analyzer.create_performance_plots()
            plt.savefig(plots_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            analyzer.create_all_depth_analysis_plots()
            plt.savefig(plots_dir / "depth_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Additional custom plots
            self._create_deadlock_comparison_plot(plots_dir)
            self._create_algorithm_heuristic_heatmap(plots_dir)
            
            print(f"📊 Plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def _create_deadlock_comparison_plot(self, output_dir: Path):
        """Create a specific comparison plot for deadlock detection impact."""
        
        df = pd.DataFrame([asdict(result) for result in self.all_results])
        solved_df = df[df['solved'] == True]
        
        if len(solved_df) == 0:
            return
        
        # Add deadlock detection flag
        solved_df = solved_df.copy()
        solved_df['has_deadlock'] = solved_df['heuristic'].str.contains('Deadlock')
        solved_df['base_heuristic'] = solved_df['heuristic'].str.replace('+Deadlock', '')
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Nodes expanded comparison
        sns.boxplot(data=solved_df, x='base_heuristic', y='nodes_expanded', 
                   hue='has_deadlock', ax=axes[0])
        axes[0].set_title('Nodes Expanded: Deadlock Detection Impact')
        axes[0].set_ylabel('Nodes Expanded')
        axes[0].set_yscale('log')
        
        # Runtime comparison
        sns.boxplot(data=solved_df, x='base_heuristic', y='runtime_seconds', 
                   hue='has_deadlock', ax=axes[1])
        axes[1].set_title('Runtime: Deadlock Detection Impact')
        axes[1].set_ylabel('Runtime (seconds)')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / "deadlock_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_algorithm_heuristic_heatmap(self, output_dir: Path):
        """Create a heatmap showing algorithm-heuristic performance."""
        
        df = pd.DataFrame([asdict(result) for result in self.all_results])
        
        # Create pivot table for success rates
        success_pivot = df.groupby(['algorithm', 'heuristic'])['solved'].mean().unstack()
        
        # Create pivot table for average nodes expanded (for solved puzzles only)
        solved_df = df[df['solved'] == True]
        if len(solved_df) > 0:
            nodes_pivot = solved_df.groupby(['algorithm', 'heuristic'])['nodes_expanded'].mean().unstack()
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Success rate heatmap
            sns.heatmap(success_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                       ax=axes[0], cbar_kws={'label': 'Success Rate'})
            axes[0].set_title('Success Rate by Algorithm and Heuristic')
            
            # Nodes expanded heatmap
            sns.heatmap(nodes_pivot, annot=True, fmt='.0f', cmap='RdYlBu_r', 
                       ax=axes[1], cbar_kws={'label': 'Avg Nodes Expanded'})
            axes[1].set_title('Average Nodes Expanded by Algorithm and Heuristic')
            
            plt.tight_layout()
            plt.savefig(output_dir / "algorithm_heuristic_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def print_summary(self):
        """Print a quick summary of all results."""
        
        if not self.all_results:
            print("No results loaded.")
            return
        
        df = pd.DataFrame([asdict(result) for result in self.all_results])
        
        print("\n" + "="*60)
        print("CLUSTER EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        
        # Overall stats
        total = len(df)
        solved = len(df[df['solved'] == True])
        print(f"Total experiments: {total}")
        print(f"Solved: {solved} ({solved/total*100:.1f}%)")
        
        # By algorithm
        print(f"\nBy Algorithm:")
        for alg in df['algorithm'].unique():
            alg_df = df[df['algorithm'] == alg]
            alg_solved = len(alg_df[alg_df['solved'] == True])
            print(f"  {alg:12} {alg_solved:3d}/{len(alg_df):3d} ({alg_solved/len(alg_df)*100:5.1f}%)")
        
        # By heuristic
        print(f"\nBy Heuristic:")
        for heur in df['heuristic'].unique():
            heur_df = df[df['heuristic'] == heur]
            heur_solved = len(heur_df[heur_df['solved'] == True])
            print(f"  {heur:20} {heur_solved:3d}/{len(heur_df):3d} ({heur_solved/len(heur_df)*100:5.1f}%)")
        
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze cluster experiment results")
    parser.add_argument('--results-dir', default='results',
                       help='Base directory containing results')
    parser.add_argument('--action', choices=['analyze', 'summary', 'plots'], 
                       default='analyze', help='Action to perform')
    
    args = parser.parse_args()
    
    analyzer = ClusterResultsAnalyzer(args.results_dir)
    
    # Load all results
    count = analyzer.load_all_results()
    
    if count == 0:
        print("No results found to analyze.")
        return
    
    if args.action == 'analyze':
        analyzer.print_summary()
        analyzer.generate_comprehensive_report()
        analyzer.create_comparison_plots()
        
    elif args.action == 'summary':
        analyzer.print_summary()
        
    elif args.action == 'plots':
        analyzer.create_comparison_plots()
    
    print(f"\n🎉 Analysis completed!")


if __name__ == "__main__":
    main() 