"""
CLI Game Player for Dots and Boxes.
Supports Human vs AI, AI vs AI, and Human vs Human.
Usage:
    python3 play.py --p1 human --p2 greedy
    python3 play.py --p1 minimax --p2 mcts --delay 0.3
    python3 play.py --rows 2 --cols 2 --p1 human --p2 random
"""

import argparse
import time
import sys
from core.board import Board
from core.game import Game
from agents import (
    RandomAgent,
    GreedyAgent,
    HumanAgent,
    RuleBasedAgent,
    MinimaxAgent,
    MCTSAgent,
    QLearningAgent
)

AGENT_MAP = {
    "human": lambda: HumanAgent("Human Player"),
    "random": lambda: RandomAgent("Random Baseline"),
    "greedy": lambda: GreedyAgent("Greedy Heuristic"),
    "rule_based": lambda: RuleBasedAgent("Rule-Based Expert (Ranjit)"),
    "minimax": lambda: MinimaxAgent("Alpha-Beta Minimax (Shanmukh)", max_depth=4),
    "mcts": lambda: MCTSAgent("Monte Carlo Tree Search (Saiyam)", iterations=250),
    "qlearning": lambda: QLearningAgent("Q-Learning RL (Naman)", epsilon=0.0)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Play Dots and Boxes in CLI")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3, help="Grid cols (default: 3)")
    parser.add_argument("--p1", type=str, default="human", choices=list(AGENT_MAP.keys()), help="Player 1 type")
    parser.add_argument("--p2", type=str, default="greedy", choices=list(AGENT_MAP.keys()), help="Player 2 type")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between AI moves (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()
    agent1 = AGENT_MAP[args.p1]()
    agent2 = AGENT_MAP[args.p2]()

    print("=" * 55)
    print("           DOTS AND BOXES - CLI MATCH            ")
    print("=" * 55)
    print(f"Board size : {args.rows}x{args.cols} ({args.rows * args.cols} boxes)")
    print(f"Player 1   : {agent1.name}")
    print(f"Player 2   : {agent2.name}")
    print("=" * 55 + "\n")

    game = Game(agent1, agent2, rows=args.rows, cols=args.cols, verbose=False)

    def on_move(board: Board, player: int, move: int):
        p_name = agent1.name if player == 1 else agent2.name
        edge_repr = board.id_to_edge(move)
        print(f"\n[Move] Player {player} ({p_name}) played Edge {move} {edge_repr}")
        print(board.render_str(show_indices=False))
        if args.delay > 0 and not isinstance(game.agents[player], HumanAgent):
            time.sleep(args.delay)

    # Initial board state
    print("Initial Board State:")
    print(game.board.render_str(show_indices=True))

    result = game.play(on_move_callback=on_move)

    print("\n" + "=" * 55)
    print("                    GAME OVER                    ")
    print("=" * 55)
    print(f"Winner       : {result.winner_name}")
    print(f"Final Score  : P1 ({result.p1_name}) = {result.scores[1]} | P2 ({result.p2_name}) = {result.scores[2]}")
    print(f"Total Moves  : {result.total_moves}")
    print(f"Game Duration: {result.duration_sec:.2f}s")
    print(f"P1 Avg Move  : {result.p1_avg_time_ms:.2f} ms")
    print(f"P2 Avg Move  : {result.p2_avg_time_ms:.2f} ms")
    print("=" * 55)


if __name__ == "__main__":
    main()
