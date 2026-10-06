"""
Multi-Grid Benchmark Suite for Dots and Boxes AI Platform.
Evaluates all agents across 2x2, 3x3, and 4x4 boards, computing cross-grid
leaderboards, decision latencies, and head-to-head performance.

Usage:
    python3 benchmark_all_grids.py --games 2
    python3 benchmark_all_grids.py --games 4 --export all_grids_results.json
"""

import argparse
import json
import time
from typing import List, Dict, Any
from agents import (
    RandomAgent,
    GreedyAgent,
    RuleBasedAgent,
    MinimaxAgent,
    MCTSAgent,
    QLearningAgent
)
from tournament import Tournament


def get_agents(rows: int, cols: int):
    """Instantiate agents configured appropriately for grid dimensions."""
    # Scale search depth / rollouts with grid size for responsiveness
    if rows == 2:
        minimax_depth = 4
        mcts_iterations = 200
    elif rows == 3:
        minimax_depth = 3
        mcts_iterations = 150
    else:  # 4x4 and above
        minimax_depth = 2
        mcts_iterations = 75

    return [
        RandomAgent("Random Baseline"),
        GreedyAgent("Greedy Heuristic"),
        RuleBasedAgent("Rule-Based Expert (Ranjit)"),
        MinimaxAgent("Alpha-Beta Minimax (Shanmukh)", max_depth=minimax_depth),
        MCTSAgent("Monte Carlo Tree Search (Saiyam)", iterations=mcts_iterations),
        QLearningAgent("Q-Learning RL (Naman)", epsilon=0.0)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Multi-Grid Benchmark Suite (2x2, 3x3, 4x4)")
    parser.add_argument("--games", type=int, default=2, help="Games per matchup per grid (default: 2, alternating sides)")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Per-move timeout in seconds")
    parser.add_argument("--export", type=str, default="all_grids_benchmark_results.json", help="Path to export JSON results")
    return parser.parse_args()


def print_cross_grid_summary(all_results: Dict[str, List[Dict[str, Any]]]):
    """Print multi-grid summary table comparing agent performance across 2x2, 3x3, and 4x4."""
    header = f"{'Agent Name':<32} | {'2x2 Win%':<9} | {'2x2 Pts':<8} | {'3x3 Win%':<9} | {'3x3 Pts':<8} | {'4x4 Win%':<9} | {'4x4 Pts':<8} | {'Avg Win%':<9}"
    sep = "=" * len(header)
    print("\n" + sep)
    print("                    CROSS-GRID MULTI-TIER BENCHMARK SUMMARY")
    print(sep)
    print(header)
    print("-" * len(header))

    agent_names = [
        "Q-Learning RL (Naman)",
        "Alpha-Beta Minimax (Shanmukh)",
        "Rule-Based Expert (Ranjit)",
        "Greedy Heuristic",
        "Monte Carlo Tree Search (Saiyam)",
        "Random Baseline"
    ]

    summary_rows = []
    for name in agent_names:
        stats_by_grid = {}
        for grid_key in ["2x2", "3x3", "4x4"]:
            grid_list = all_results.get(grid_key, [])
            match = next((item for item in grid_list if item["name"] == name), None)
            if match:
                stats_by_grid[grid_key] = match
            else:
                stats_by_grid[grid_key] = {"win_rate": 0.0, "avg_points": 0.0}

        w2 = stats_by_grid["2x2"]["win_rate"]
        p2 = stats_by_grid["2x2"]["avg_points"]
        w3 = stats_by_grid["3x3"]["win_rate"]
        p3 = stats_by_grid["3x3"]["avg_points"]
        w4 = stats_by_grid["4x4"]["win_rate"]
        p4 = stats_by_grid["4x4"]["avg_points"]
        avg_w = (w2 + w3 + w4) / 3.0

        summary_rows.append({
            "name": name,
            "w2": w2, "p2": p2,
            "w3": w3, "p3": p3,
            "w4": w4, "p4": p4,
            "avg_w": avg_w
        })

    summary_rows.sort(key=lambda x: x["avg_w"], reverse=True)

    for r in summary_rows:
        print(
            f"{r['name']:<32} | "
            f"{r['w2']:>7.1f}% | {r['p2']:>6.2f}  | "
            f"{r['w3']:>7.1f}% | {r['p3']:>6.2f}  | "
            f"{r['w4']:>7.1f}% | {r['p4']:>6.2f}  | "
            f"{r['avg_w']:>7.1f}%"
        )
    print(sep + "\n")


def main():
    args = parse_args()

    grids = [
        (2, 2, "2x2 (4 boxes, 12 edges)"),
        (3, 3, "3x3 (9 boxes, 24 edges)"),
        (4, 4, "4x4 (16 boxes, 40 edges)")
    ]

    suite_start = time.perf_counter()
    all_grid_results = {}

    print("\n" + "#" * 65)
    print("      DOTS AND BOXES: TRIPLE-GRID COMPREHENSIVE BENCHMARK      ")
    print("      Grids: 2x2, 3x3, 4x4 | Games per matchup: " + str(args.games))
    print("#" * 65)

    for rows, cols, desc in grids:
        print(f"\n>>> RUNNING BENCHMARK ON {desc} <<<")
        agents = get_agents(rows, cols)

        tournament = Tournament(
            agents=agents,
            games_per_matchup=args.games,
            rows=rows,
            cols=cols,
            time_limit_sec=args.time_limit,
            verbose=False
        )

        res = tournament.run()
        grid_key = f"{rows}x{cols}"
        all_grid_results[grid_key] = res["leaderboard"]

    suite_elapsed = time.perf_counter() - suite_start
    print(f"All benchmarks finished in {suite_elapsed:.2f} seconds.")

    # Cross-grid summary
    print_cross_grid_summary(all_grid_results)

    # Export to JSON
    if args.export:
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "games_per_matchup": args.games,
            "suite_duration_sec": suite_elapsed,
            "results_by_grid": all_grid_results
        }
        with open(args.export, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Full benchmark data exported to {args.export}\n")


if __name__ == "__main__":
    main()
