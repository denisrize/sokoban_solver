import heapq
import time
import tracemalloc
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

@dataclass
class Position:
    """Represents a 2D position."""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

@dataclass
class GameState:
    """Represents the complete state of a Sokoban puzzle."""
    player: Position
    boxes: Set[Position]
    goals: Set[Position]
    walls: Set[Position]
    width: int
    height: int
    moves: int = 0
    path: List[str] = None
    puzzle_name: str = None

    def __post_init__(self):
        if self.path is None:
            self.path = []
    
    def __hash__(self):
        return hash((self.player, frozenset(self.boxes)))
    
    def __eq__(self, other):
        return (self.player == other.player and 
                self.boxes == other.boxes)
    
    def is_goal_state(self) -> bool:
        """Check if all boxes are on goal positions."""
        return self.boxes == self.goals
    
    def __str__(self) -> str:
        """Return string representation of the current game state."""
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = Position(x, y)
                
                if pos in self.walls:
                    row.append('#')
                elif pos == self.player:
                    if pos in self.goals:
                        row.append('+')
                    else:
                        row.append('@')
                elif pos in self.boxes:
                    if pos in self.goals:
                        row.append('*')
                    else:
                        row.append('$')
                elif pos in self.goals:
                    row.append('.')
                else:
                    row.append(' ')
            result.append(''.join(row))
        return '\n'.join(result)
    
    def get_valid_moves(self) -> List[Tuple[str, 'GameState']]:
        """Get all valid moves from current state."""
        moves = []
        directions = [('U', Position(0, -1)), ('D', Position(0, 1)), 
                     ('L', Position(-1, 0)), ('R', Position(1, 0))]
        
        for direction, delta in directions:
            new_player = self.player + delta
            
            # Check if new position is within bounds and not a wall
            if (0 <= new_player.x < self.width and 
                0 <= new_player.y < self.height and 
                new_player not in self.walls):
                
                # Check if there's a box at the new position
                if new_player in self.boxes:
                    # Calculate where the box would move
                    new_box_pos = new_player + delta
                    
                    # Check if box can be pushed
                    if (0 <= new_box_pos.x < self.width and 
                        0 <= new_box_pos.y < self.height and 
                        new_box_pos not in self.walls and 
                        new_box_pos not in self.boxes):
                        
                        # Create new state with box moved efficiently
                        new_state = self._create_state_with_box_push(new_player, new_box_pos, direction)
                        moves.append((direction, new_state))
                else:
                    # Create new state with just player move efficiently
                    new_state = self._create_state_player_move(new_player, direction)
                    moves.append((direction, new_state))
        
        return moves
    
    def _create_state_player_move(self, new_player: Position, direction: str) -> 'GameState':
        """Efficiently create new state for player-only moves (no deepcopy)."""
        return GameState(
            player=new_player,
            boxes=self.boxes.copy(),  # Only copy the mutable set
            goals=self.goals,         # Share immutable reference
            walls=self.walls,         # Share immutable reference  
            width=self.width,
            height=self.height,
            moves=self.moves + 1,
            path=self.path + [direction],
            puzzle_name=self.puzzle_name
        )
    
    def _create_state_with_box_push(self, new_player: Position, new_box_pos: Position, direction: str) -> 'GameState':
        """Efficiently create new state for box-pushing moves (no deepcopy)."""
        # Efficiently update boxes set
        new_boxes = self.boxes.copy()
        new_boxes.remove(new_player)  # Remove box from old position
        new_boxes.add(new_box_pos)    # Add box to new position
        
        return GameState(
            player=new_player,
            boxes=new_boxes,
            goals=self.goals,         # Share immutable reference
            walls=self.walls,         # Share immutable reference
            width=self.width,
            height=self.height,
            moves=self.moves + 1,
            path=self.path + [direction],
            puzzle_name=self.puzzle_name
        )

class HeuristicFunctions:
    """Collection of heuristic functions for Sokoban."""
    
    @staticmethod
    def manhattan_distance(state: GameState) -> int:
        """Manhattan distance heuristic: sum of minimum distances from each box to closest goal."""
        if not state.boxes:
            return 0
        
        total_distance = 0
        for box in state.boxes:
            min_dist = float('inf')
            for goal in state.goals:
                dist = abs(box.x - goal.x) + abs(box.y - goal.y)
                min_dist = min(min_dist, dist)
            total_distance += min_dist
        
        return total_distance
    
    @staticmethod
    def minimum_cost_matching(state: GameState) -> int:
        """Hungarian algorithm for optimal box-goal assignment."""
        if not state.boxes or not state.goals:
            return 0
        
        boxes = list(state.boxes)
        goals = list(state.goals)
        
        # Create cost matrix
        cost_matrix = np.zeros((len(boxes), len(goals)))
        for i, box in enumerate(boxes):
            for j, goal in enumerate(goals):
                cost_matrix[i][j] = abs(box.x - goal.x) + abs(box.y - goal.y)
        
        # Pad matrix if needed (more boxes than goals or vice versa)
        if len(boxes) != len(goals):
            max_size = max(len(boxes), len(goals))
            padded_matrix = np.full((max_size, max_size), cost_matrix.max() + 1)
            padded_matrix[:len(boxes), :len(goals)] = cost_matrix
            cost_matrix = padded_matrix
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return int(cost_matrix[row_indices, col_indices].sum())
    
    @staticmethod
    def detect_corner_deadlocks(state: GameState) -> bool:
        """Detect corner deadlock states (boxes in corners without goals).
        For Example:
        #####
        #$   #
        #@   #
        # $  .#
        # . ##
        #####
        """
        directions = {'D': Position(0, 1), 'U': Position(0, -1), 'R': Position(1, 0), 'L': Position(-1, 0)}
        
        for box in state.boxes:
            if box in state.goals:
                continue  # Box is on goal, not a deadlock
            
            # Check if box is in a corner
            adjacent_walls = set()
            for dir_name, direction in directions.items():
                adjacent_pos = box + direction
                if adjacent_pos in state.walls:
                    adjacent_walls.add(dir_name)
            
            # If box has 2 adjacent walls and is not on goal
            if len(adjacent_walls) >= 2:
                if {'D', 'R'}.issubset(adjacent_walls):
                    return True
                elif {'U', 'L'}.issubset(adjacent_walls):
                    return True
                elif {'U', 'R'}.issubset(adjacent_walls):
                    return True
                elif {'D', 'L'}.issubset(adjacent_walls):
                    return True
        
        return False
    
    @staticmethod
    def detect_edge_deadlocks(state: GameState) -> bool:
        """Detects if a box is truly stuck along a wall with no escape or reachable goals.
        For Example:
        #####
        #@   #
        #$   #
        # $  .#
        # . ##
        #####
        """
        for box in state.boxes:
            if box in state.goals:
                continue
                
            # Check if box is against an edge and truly trapped
            if HeuristicFunctions._is_edge_deadlocked(state, box):
                return True
        return False

    @staticmethod
    def detect_adjacent_boxes_block_deadlocks(state: GameState) -> bool:
        """Detects if adjacent boxes are frozen by walls in a way that prevents reaching goals.
            For Example:
            #####
            #@   #
            #$   #
            #$   .#
            #.  ##
            #####
                """
        for box in state.boxes:
            if box in state.goals:
                continue
                
            # Check for adjacent boxes
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                neighbor = Position(box.x+dx, box.y+dy)
                if neighbor in state.boxes:
                    # Found adjacent boxes not on goals
                    # Check if they're constrained by walls
                    if HeuristicFunctions._are_boxes_frozen_by_walls(state, box, neighbor):
                        return True
        return False

    @staticmethod
    def _is_edge_deadlocked(state: GameState, box: Position) -> bool:
        """Check if a specific box is deadlocked on an edge."""
        # Check vertical edges (left/right walls)
        if box.x == 1 or box.x == state.width - 2:
            # Box is trapped against edge, check if there are reachable goals along this edge
            return not HeuristicFunctions._has_reachable_goals_on_vertical_edge(state, box)
        
        # Check horizontal edges (top/bottom walls)
        if box.y == 1 or box.y == state.height - 2:
            # Box is trapped against edge, check if there are reachable goals along this edge
            return not HeuristicFunctions._has_reachable_goals_on_horizontal_edge(state, box)
        
        return False  # Box is not on an edge
    
    @staticmethod
    def _has_reachable_goals_on_vertical_edge(state: GameState, box: Position) -> bool:
        """Check if there are any goals reachable by sliding along the vertical edge."""
        edge_x = box.x 
        total_boxes = 1
        total_goals = 0
        # Check up direction
        y = box.y - 1
        while y > 0:
            pos = Position(edge_x, y)
            if pos in state.walls:
                break
            if pos in state.boxes:
                total_boxes += 1
            if pos in state.goals:
                total_goals += 1
            y -= 1
        
        # Check down direction
        y = box.y + 1
        while y < state.height - 1:
            pos = Position(edge_x, y)
            if pos in state.walls:
                break
            if pos in state.boxes:
                total_boxes += 1
            if pos in state.goals:
                total_goals += 1
            y += 1
        
        return total_boxes == total_goals
    
    @staticmethod
    def _has_reachable_goals_on_horizontal_edge(state: GameState, box: Position) -> bool:
        """Check if there are any goals reachable by sliding along the horizontal edge."""
        edge_y = box.y
        total_boxes = 1
        total_goals = 0
        # Check left direction
        x = box.x - 1
        while x >= 0:
            pos = Position(x, edge_y)
            if pos in state.walls:
                break
            if pos in state.boxes:
                total_boxes += 1
            if pos in state.goals:
                total_goals += 1
            x -= 1
        
        # Check right direction
        x = box.x + 1
        while x < state.width - 1:
            pos = Position(x, edge_y)
            if pos in state.walls:
                break
            if pos in state.boxes:
                total_boxes += 1
            if pos in state.goals:
                total_goals += 1
            x += 1
        
        return total_boxes == total_goals

    @staticmethod
    def _are_boxes_frozen_by_walls(state: GameState, box1: Position, box2: Position) -> bool:
        """Check if two adjacent boxes are frozen by wall constraints."""
        # Check if boxes are against walls that prevent separation
        # For vertical adjacency - check left/right walls
        if box1.x == box2.x:  # Vertically adjacent
            left_blocked = (Position(box1.x-1, box1.y) in state.walls and 
                        Position(box2.x-1, box2.y) in state.walls)
            right_blocked = (Position(box1.x+1, box1.y) in state.walls and 
                            Position(box2.x+1, box2.y) in state.walls)
            return left_blocked or right_blocked
        
        # For horizontal adjacency - check top/bottom walls  
        elif box1.y == box2.y:  # Horizontally adjacent
            top_blocked = (Position(box1.x, box1.y-1) in state.walls and 
                        Position(box2.x, box2.y-1) in state.walls)
            bottom_blocked = (Position(box1.x, box1.y+1) in state.walls and 
                            Position(box2.x, box2.y+1) in state.walls)
            return top_blocked or bottom_blocked
        
        return False

    @staticmethod
    def detect_advanced_deadlocks(state: GameState) -> bool:
        """Unified advanced deadlock detection."""
        return (
            HeuristicFunctions.detect_corner_deadlocks(state) or
            HeuristicFunctions.detect_edge_deadlocks(state) or
            HeuristicFunctions.detect_adjacent_boxes_block_deadlocks(state)
        )



class SearchAlgorithms:
    """Collection of search algorithms for solving Sokoban."""
    
    def __init__(self):
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_queue_size = 0
        # AWA* specific statistics
        self.awa_iterations = []
        self.awa_best_solution = None
        self.awa_total_peak_memory_mb = 0
    
    def reset_stats(self):
        """Reset search statistics."""
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_queue_size = 0
        self.awa_iterations = []
        self.awa_best_solution = None
        self.awa_total_peak_memory_mb = 0
    
    def a_star(self, initial_state: GameState, heuristic_func, use_deadlock_detection=True) -> Optional[GameState]:
        """A* search algorithm."""
        self.reset_stats()
        
        counter = 0  # Tie-breaker to avoid GameState comparison
        open_list = [(heuristic_func(initial_state), 0, counter, initial_state)]
        closed_set = set()
        g_scores = {initial_state: 0}
        
        heapq.heapify(open_list)
        
        while open_list:
            self.max_queue_size = max(self.max_queue_size, len(open_list))
            
            _, g_score, _, current_state = heapq.heappop(open_list)
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            self.nodes_expanded += 1
            
            if current_state.is_goal_state():
                return current_state
            
            for move, next_state in current_state.get_valid_moves():
                if next_state in closed_set:
                    continue
                
                # Prune deadlocked states completely (if enabled)
                if use_deadlock_detection and HeuristicFunctions.detect_advanced_deadlocks(next_state):
                    continue
                
                tentative_g = g_score + 1
                
                if next_state not in g_scores or tentative_g < g_scores[next_state]:
                    g_scores[next_state] = tentative_g
                    f_score = tentative_g + heuristic_func(next_state)
                    counter += 1
                    heapq.heappush(open_list, (f_score, tentative_g, counter, next_state))
                    self.nodes_generated += 1
        
        return None
    
    def ida_star(self, initial_state: GameState, heuristic_func, use_deadlock_detection=True) -> Optional[GameState]:
        """Iterative Deepening A* search algorithm."""
        self.reset_stats()
        
        def search(state: GameState, g_score: int, threshold: int, path: Set[GameState]) -> Tuple[Optional[GameState], int]:
            f_score = g_score + heuristic_func(state)
            
            if f_score > threshold:
                return None, f_score
            
            if state.is_goal_state():
                return state, f_score
            
            self.nodes_expanded += 1
            min_threshold = float('inf')
            
            for move, next_state in state.get_valid_moves():
                if next_state not in path:
                    # Prune deadlocked states completely (if enabled)
                    if use_deadlock_detection and HeuristicFunctions.detect_advanced_deadlocks(next_state):
                        continue
                    
                    self.nodes_generated += 1
                    path.add(next_state)
                    result, new_threshold = search(next_state, g_score + 1, threshold, path)
                    path.remove(next_state)
                    
                    if result is not None:
                        return result, new_threshold
                    
                    min_threshold = min(min_threshold, new_threshold)
            
            return None, min_threshold
        
        threshold = heuristic_func(initial_state)
        
        while True:
            path = {initial_state}
            result, new_threshold = search(initial_state, 0, threshold, path)
            
            if result is not None:
                return result
            
            if new_threshold == float('inf'):
                return None
            
            threshold = new_threshold
    
    def ida_star_optimized(self, initial_state: GameState, heuristic_func, use_deadlock_detection=True, 
                          max_time_seconds=None, results_dir="ida_star_results") -> Optional[GameState]:
        """
        Optimized IDA* with move ordering and progress reporting.
        
        Optimizations:
        - Move ordering by heuristic value (explores best moves first)
        - Efficient state creation (no deepcopy)
        - Progress reporting saved to results directory
        - Time limits to prevent infinite search
        
        Maintains IDA* optimality guarantees.
        """
        self.reset_stats()
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        # File name should include heuristic name, use_deadlock_detection, and puzzle name
        progress_file = os.path.join(results_dir, f"ida_star_progress_{heuristic_func.__name__}_{use_deadlock_detection}_{initial_state.puzzle_name}.json")
        
        # Initialize progress tracking
        progress_data = {
            "start_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "algorithm": "IDA* Optimized",
            "heuristic": heuristic_func.__name__ if hasattr(heuristic_func, '__name__') else "unknown",
            "use_deadlock_detection": use_deadlock_detection,
            "max_time_seconds": max_time_seconds,
            "iterations": [],
            "final_result": None
        }
        
        start_time = time.time()
        
        def search(state: GameState, g_score: int, threshold: int, path: Set[GameState]) -> Tuple[Optional[GameState], int]:
            # Time limit check
            if max_time_seconds is not None and time.time() - start_time > max_time_seconds:
                return None, float('inf')
            
            f_score = g_score + heuristic_func(state)
            
            if f_score > threshold:
                return None, f_score
            
            if state.is_goal_state():
                return state, f_score
            
            self.nodes_expanded += 1
            min_threshold = float('inf')
            
            # Get all valid moves
            moves = state.get_valid_moves()
            
            # OPTIMIZATION 1: Move ordering by heuristic value
            # Calculate heuristic for each move and sort (best first)
            move_evaluations = []
            for move, next_state in moves:
                if next_state not in path:
                    # Early deadlock detection
                    if use_deadlock_detection and HeuristicFunctions.detect_advanced_deadlocks(next_state):
                        continue
                    
                    h_value = heuristic_func(next_state)
                    move_evaluations.append((h_value, move, next_state))
            
            # Sort by heuristic value (lower is better - explores most promising moves first)
            move_evaluations.sort(key=lambda x: x[0])
            
            # Explore moves in heuristic order
            for h_value, move, next_state in move_evaluations:
                self.nodes_generated += 1
                path.add(next_state)
                result, new_threshold = search(next_state, g_score + 1, threshold, path)
                path.remove(next_state)
                
                if result is not None:
                    return result, new_threshold
                
                min_threshold = min(min_threshold, new_threshold)
            
            return None, min_threshold
        
        # Start with heuristic value as initial threshold
        threshold = heuristic_func(initial_state)
        iteration = 0
        
        print(f"Starting Optimized IDA* with initial threshold: {threshold}")
        print(f"Progress will be saved to: {progress_file}")
        
        while threshold < float('inf') and (max_time_seconds is None or time.time() - start_time < max_time_seconds):
            iteration += 1
            iteration_start_time = time.time()
            prev_nodes_expanded = self.nodes_expanded
            prev_nodes_generated = self.nodes_generated
            
            # Run search iteration
            path = {initial_state}
            result, new_threshold = search(initial_state, 0, threshold, path)
            
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_nodes_expanded = self.nodes_expanded - prev_nodes_expanded
            iteration_nodes_generated = self.nodes_generated - prev_nodes_generated
            
            # Record iteration progress
            iteration_data = {
                "iteration": iteration,
                "threshold": threshold,
                "duration_seconds": round(iteration_duration, 3),
                "nodes_expanded": iteration_nodes_expanded,
                "nodes_generated": iteration_nodes_generated,
                "total_nodes_expanded": self.nodes_expanded,
                "total_nodes_generated": self.nodes_generated,
                "cumulative_time": round(time.time() - start_time, 3),
                "solution_found": result is not None,
                "next_threshold": new_threshold if new_threshold != float('inf') else None
            }
            
            progress_data["iterations"].append(iteration_data)
            
            # Save progress after each iteration
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            # Print iteration summary
            print(f"Iteration {iteration:3d}: threshold={threshold:4d}, "
                  f"time={iteration_duration:6.2f}s, "
                  f"nodes_exp={iteration_nodes_expanded:8,}, "
                  f"total_time={time.time() - start_time:7.1f}s")
            
            if result is not None:
                print(f"✓ SOLUTION FOUND in {iteration} iterations!")
                print(f"  Solution length: {result.moves} moves")
                print(f"  Total time: {time.time() - start_time:.2f} seconds")
                print(f"  Total nodes expanded: {self.nodes_expanded:,}")
                
                # Record final result
                progress_data["final_result"] = {
                    "solution_found": True,
                    "solution_length": result.moves,
                    "total_iterations": iteration,
                    "total_time_seconds": round(time.time() - start_time, 3),
                    "total_nodes_expanded": self.nodes_expanded,
                    "total_nodes_generated": self.nodes_generated,
                    "solution_path": result.path
                }
                
                # Final save with solution
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                
                return result
            
            if new_threshold == float('inf'):
                print("✗ No solution exists (search space exhausted)")
                break
                
            threshold = new_threshold
        
        # Handle timeout or no solution
        if max_time_seconds is not None and time.time() - start_time >= max_time_seconds:
            print(f"⚠ Search terminated due to time limit ({max_time_seconds}s)")
        
        print(f"Search completed: {iteration} iterations, {self.nodes_expanded:,} nodes expanded")
        
        # Record final result (no solution)
        progress_data["final_result"] = {
            "solution_found": False,
            "total_iterations": iteration,
            "total_time_seconds": round(time.time() - start_time, 3),
            "total_nodes_expanded": self.nodes_expanded,
            "total_nodes_generated": self.nodes_generated,
            "termination_reason": "timeout" if max_time_seconds is not None and time.time() - start_time >= max_time_seconds else "no_solution"
        }
        
        # Final save
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        return None
    
    def best_first_search(self, initial_state: GameState, heuristic_func, use_deadlock_detection=True) -> Optional[GameState]:
        """Greedy Best-First search algorithm."""
        self.reset_stats()
        
        counter = 0  # Tie-breaker to avoid GameState comparison
        open_list = [(heuristic_func(initial_state), counter, initial_state)]
        closed_set = set()
        
        heapq.heapify(open_list)
        
        while open_list:
            self.max_queue_size = max(self.max_queue_size, len(open_list))
            
            _, _, current_state = heapq.heappop(open_list)
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            self.nodes_expanded += 1
            
            if current_state.is_goal_state():
                return current_state
            
            for move, next_state in current_state.get_valid_moves():
                if next_state not in closed_set:
                    # Prune deadlocked states completely (if enabled)
                    if use_deadlock_detection and HeuristicFunctions.detect_advanced_deadlocks(next_state):
                        continue
                    
                    counter += 1
                    heapq.heappush(open_list, (heuristic_func(next_state), counter, next_state))
                    self.nodes_generated += 1
        
        return None

    def anytime_weighted_a_star(self, initial_state: GameState, heuristic_func, use_deadlock_detection=True) -> Optional[GameState]:
        """
        Anytime Weighted A* search algorithm with solution bounding.
        
        Uses geometric progression weights: [4.0, 2.4, 1.44, 0.864]
        Each iteration improves solution quality while maintaining speed benefit.
        
        Returns the best solution found across all iterations.
        Detailed iteration results are stored in self.awa_iterations.
        """
        self.reset_stats()
        
        # Geometric progression weights: start=4.0, ratio=0.6, iterations=4
        weights = [4.0, 2.4, 1.44]
        best_solution = None
        best_cost = float('inf')
        total_nodes_expanded = 0
        total_nodes_generated = 0
        overall_peak_memory = 0
        
        print(f"AWA* starting with weights: {weights}")
        
        for iteration, weight in enumerate(weights):
            print(f"\n--- AWA* Iteration {iteration + 1}/4 (weight={weight:.3f}) ---")
            
            # Start timing and memory tracking for this iteration
            iteration_start_time = time.time()
            tracemalloc.start()
            
            # Run bounded weighted A* for this iteration
            solution = self._bounded_weighted_a_star(
                initial_state, 
                heuristic_func, 
                weight, 
                bound=best_cost,
                use_deadlock_detection=use_deadlock_detection
            )
            
            # Get memory peak and stop tracking
            _, iteration_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            overall_peak_memory = max(overall_peak_memory, iteration_peak)
            
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            
            # Track iteration statistics
            iteration_stats = {
                'iteration': iteration + 1,
                'weight': weight,
                'nodes_expanded': self.nodes_expanded - total_nodes_expanded,
                'nodes_generated': self.nodes_generated - total_nodes_generated,
                'runtime_seconds': iteration_time,
                'solved': solution is not None,
                'solution_length': solution.moves if solution else 0,
                'bound_used': best_cost if best_cost != float('inf') else None,
                'memory_peak_mb': iteration_peak / 1024 / 1024
            }
            
            # Update totals
            total_nodes_expanded = self.nodes_expanded
            total_nodes_generated = self.nodes_generated
            
            if solution and solution.moves < best_cost:
                best_solution = solution
                best_cost = solution.moves
                iteration_stats['improved'] = True
                print(f"✓ NEW BEST SOLUTION: {solution.moves} moves")
                print(f"  Nodes expanded: {iteration_stats['nodes_expanded']:,}")
                print(f"  Time: {iteration_time:.2f}s")
            else:
                iteration_stats['improved'] = False
                if solution:
                    print(f"✓ Found solution: {solution.moves} moves (not better than {best_cost})")
                else:
                    print(f"✗ No solution found within bound {best_cost}")
                print(f"  Nodes expanded: {iteration_stats['nodes_expanded']:,}")
                print(f"  Time: {iteration_time:.2f}s")
            
            self.awa_iterations.append(iteration_stats)
            
            # Early termination if no improvement and getting expensive
            if iteration > 0 and not iteration_stats['improved'] and iteration_time > 60:
                print(f"⚠ Early termination: no improvement and iteration taking too long")
                break
        
        # Store best solution for external access
        self.awa_best_solution = best_solution
        self.awa_total_peak_memory_mb = overall_peak_memory / 1024 / 1024
        
        print(f"\n--- AWA* SUMMARY ---")
        print(f"Best solution: {best_cost} moves" if best_solution else "No solution found")
        print(f"Total nodes expanded: {self.nodes_expanded:,}")
        print(f"Total iterations: {len(self.awa_iterations)}")
        
        return best_solution
    
    def _bounded_weighted_a_star(self, initial_state: GameState, heuristic_func, weight: float, 
                               bound: float, use_deadlock_detection: bool) -> Optional[GameState]:
        """
        Weighted A* search with solution bounding.
        
        Args:
            weight: Heuristic weight (w > 1 for suboptimal, w < 1 for super-optimal)
            bound: Upper bound on solution cost for pruning
        """
        counter = 0
        open_list = []
        closed_set = set()
        g_scores = {initial_state: 0}
        
        # Initial state f-value
        initial_f = 0 + weight * heuristic_func(initial_state)
        heapq.heappush(open_list, (initial_f, 0, counter, initial_state))
        
        iteration_nodes_expanded = 0
        iteration_nodes_generated = 0
        
        while open_list:
            self.max_queue_size = max(self.max_queue_size, len(open_list))
            
            f_score, g_score, _, current_state = heapq.heappop(open_list)
            
            # Solution bounding: prune if f-score exceeds bound
            if f_score >= bound:
                continue
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            self.nodes_expanded += 1
            iteration_nodes_expanded += 1
            
            if current_state.is_goal_state():
                return current_state
            
            for move, next_state in current_state.get_valid_moves():
                if next_state in closed_set:
                    continue
                
                # Deadlock detection
                if use_deadlock_detection and HeuristicFunctions.detect_advanced_deadlocks(next_state):
                    continue
                
                tentative_g = g_score + 1
                
                # Solution bounding: prune if g + h >= bound
                h_value = heuristic_func(next_state)
                if tentative_g + h_value >= bound:
                    continue
                
                if next_state not in g_scores or tentative_g < g_scores[next_state]:
                    g_scores[next_state] = tentative_g
                    f_score = tentative_g + weight * h_value
                    
                    # Final bound check before adding to queue
                    if f_score < bound:
                        counter += 1
                        heapq.heappush(open_list, (f_score, tentative_g, counter, next_state))
                        self.nodes_generated += 1
                        iteration_nodes_generated += 1
        
        return None

    @staticmethod
    def analyze_ida_star_progress(progress_file: str):
        """
        Analyze and display progress data from a saved IDA* run.
        
        Args:
            progress_file: Path to the JSON progress file
        """
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Progress file not found: {progress_file}")
            return
        except json.JSONDecodeError:
            print(f"Invalid JSON in progress file: {progress_file}")
            return
        
        print("=" * 60)
        print("IDA* PROGRESS ANALYSIS")
        print("=" * 60)
        
        # Basic info
        print(f"Algorithm: {data.get('algorithm', 'N/A')}")
        print(f"Start time: {data.get('start_time', 'N/A')}")
        print(f"Heuristic: {data.get('heuristic', 'N/A')}")
        print(f"Deadlock detection: {data.get('use_deadlock_detection', 'N/A')}")
        print(f"Max time limit: {data.get('max_time_seconds', 'N/A')} seconds")
        
        iterations = data.get('iterations', [])
        final_result = data.get('final_result', {})
        
        if not iterations:
            print("\nNo iterations recorded.")
            return
        
        print(f"\nTotal iterations: {len(iterations)}")
        
        # Final result summary
        if final_result.get('solution_found'):
            print(f"✓ SOLUTION FOUND!")
            print(f"  Solution length: {final_result.get('solution_length')} moves")
            print(f"  Total time: {final_result.get('total_time_seconds'):.2f} seconds")
            print(f"  Total nodes expanded: {final_result.get('total_nodes_expanded'):,}")
        else:
            print(f"✗ NO SOLUTION FOUND")
            print(f"  Reason: {final_result.get('termination_reason', 'unknown')}")
            print(f"  Total time: {final_result.get('total_time_seconds', 0):.2f} seconds")
            print(f"  Total nodes expanded: {final_result.get('total_nodes_expanded', 0):,}")
        
        # Iteration breakdown
        print(f"\n{'Iter':<4} | {'Threshold':<9} | {'Time(s)':<8} | {'Nodes Exp':<10} | {'Cumulative':<11} | {'Status':<10}")
        print("-" * 70)
        
        for iteration in iterations:
            iter_num = iteration.get('iteration', 0)
            threshold = iteration.get('threshold', 0)
            duration = iteration.get('duration_seconds', 0)
            nodes_exp = iteration.get('nodes_expanded', 0)
            cumulative = iteration.get('cumulative_time', 0)
            status = "SOLVED" if iteration.get('solution_found') else "continue"
            
            print(f"{iter_num:<4} | {threshold:<9} | {duration:<8.2f} | {nodes_exp:<10,} | {cumulative:<11.1f} | {status:<10}")
        
        # Performance metrics
        total_time = final_result.get('total_time_seconds', 0)
        total_nodes = final_result.get('total_nodes_expanded', 0)
        
        if total_time > 0:
            nodes_per_second = total_nodes / total_time
            print(f"\nPerformance Metrics:")
            print(f"  Nodes per second: {nodes_per_second:,.0f}")
            print(f"  Average iteration time: {total_time / len(iterations):.2f} seconds")
        
        # Threshold progression
        thresholds = [iter_data.get('threshold', 0) for iter_data in iterations]
        if len(thresholds) > 1:
            print(f"\nThreshold Progression:")
            print(f"  Initial: {thresholds[0]}")
            print(f"  Final: {thresholds[-1]}")
            print(f"  Growth rate: {thresholds[-1] / thresholds[0]:.2f}x")
        
        print(f"\nProgress file: {progress_file}")
    
    @staticmethod
    def list_ida_star_results(results_dir="ida_star_results"):
        """List all saved IDA* progress files in the results directory."""
        if not os.path.exists(results_dir):
            print(f"Results directory does not exist: {results_dir}")
            return []
        
        progress_files = [f for f in os.listdir(results_dir) if f.startswith("ida_star_progress_") and f.endswith(".json")]
        progress_files.sort(reverse=True)  # Most recent first
        
        if not progress_files:
            print(f"No IDA* progress files found in {results_dir}")
            return []
        
        print(f"IDA* Progress Files in {results_dir}:")
        print("-" * 50)
        
        for i, filename in enumerate(progress_files, 1):
            file_path = os.path.join(results_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                final_result = data.get('final_result', {})
                solved = "✓" if final_result.get('solution_found') else "✗"
                solution_len = final_result.get('solution_length', 'N/A')
                total_time = final_result.get('total_time_seconds', 0)
                iterations = len(data.get('iterations', []))
                
                print(f"{i:2d}. {filename}")
                print(f"    {solved} Solution: {solution_len} moves, {total_time:.1f}s, {iterations} iterations")
                
            except (json.JSONDecodeError, KeyError):
                print(f"{i:2d}. {filename} (corrupted)")
        
        return [os.path.join(results_dir, f) for f in progress_files]

class SokobanPuzzle:
    """Sokoban puzzle representation and utilities."""
    
    def __init__(self, puzzle_string: str, puzzle_name: str):
        """Initialize puzzle from string representation."""
        lines = [line for line in puzzle_string.strip().split('\n') if line]
        self.height = len(lines)
        self.width = max(len(line) for line in lines)
        self.puzzle_name = puzzle_name
        self.player = None
        self.boxes = set()
        self.goals = set()
        self.walls = set()
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                pos = Position(x, y)
                
                if char == '#':
                    self.walls.add(pos)
                elif char == '@':
                    self.player = pos
                elif char == '$':
                    self.boxes.add(pos)
                elif char == '.':
                    self.goals.add(pos)
                elif char == '*':  # Box on goal
                    self.boxes.add(pos)
                    self.goals.add(pos)
                elif char == '+':  # Player on goal
                    self.player = pos
                    self.goals.add(pos)
    
    def get_initial_state(self) -> GameState:
        """Get the initial game state."""
        return GameState(
            player=self.player,
            boxes=self.boxes.copy(),
            goals=self.goals.copy(),
            walls=self.walls.copy(),
            width=self.width,
            height=self.height,
            puzzle_name=self.puzzle_name
        )
    
    def print_state(self, state: GameState):
        """Print the current state of the puzzle."""
        for y in range(self.height):
            for x in range(self.width):
                pos = Position(x, y)
                
                if pos in state.walls:
                    print('#', end='')
                elif pos == state.player:
                    if pos in state.goals:
                        print('+', end='')
                    else:
                        print('@', end='')
                elif pos in state.boxes:
                    if pos in state.goals:
                        print('*', end='')
                    else:
                        print('$', end='')
                elif pos in state.goals:
                    print('.', end='')
                else:
                    print(' ', end='')
            print()
        print()