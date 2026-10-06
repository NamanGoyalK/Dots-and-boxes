"""
Tournament & Benchmark CLI Runner for Dots and Boxes.
Usage:
    python3 run_tournament.py --games 10 --rows 3 --cols 3
    python3 run_tournament.py --games 2 --quick
"""

import argparse
from agents import (
    RandomAgent,
    GreedyAgent,
    RuleBasedAgent,
    MinimaxAgent,
    MCTSAgent,
    QLearningAgent
)
from tournament import Tournament


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dots and Boxes Tournament")
    parser.add_argument("--rows", type=int, default=3, help="Grid box rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3, help="Grid box cols (default: 3)")
    parser.add_argument("--games", type=int, default=6, help="Games per matchup (default: 6)")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Per-move timeout in seconds")
    parser.add_argument("--quick", action="store_true", help="Quick mode (2x2 board, 2 games per matchup)")
    parser.add_argument("--export", type=str, default="tournament_results.json", help="Path to export JSON results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    rows = 2 if args.quick else args.rows
    cols = 2 if args.quick else args.cols
    games = 2 if args.quick else args.games

    # Instantiate agents
    agents = [
        RandomAgent("Random Baseline"),
        GreedyAgent("Greedy Heuristic"),
        RuleBasedAgent("Rule-Based Expert (Ranjit)"),
        MinimaxAgent("Alpha-Beta Minimax (Shanmukh)", max_depth=3 if rows >= 3 else 4),
        MCTSAgent("Monte Carlo Tree Search (Saiyam)", iterations=150),
        QLearningAgent("Q-Learning RL (Naman)", epsilon=0.0)
    ]

    tournament = Tournament(
        agents=agents,
        games_per_matchup=games,
        rows=rows,
        cols=cols,
        time_limit_sec=args.time_limit,
        verbose=False
    )

    results = tournament.run()
    if args.export:
        tournament.export_json(args.export)


if __name__ == "__main__":
    main()
