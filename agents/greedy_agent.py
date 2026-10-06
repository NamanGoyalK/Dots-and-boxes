"""
Greedy Agent - Baseline rule-following AI.
1. Immediately captures any box that has 3 edges.
2. Otherwise, avoids moves that create a 3rd edge (giving away a box).
3. If forced to give away a box, picks the edge minimizing immediate damage.
"""

import random
from typing import Union, Tuple
from core.base_agent import BaseAgent
from core.board import Board


class GreedyAgent(BaseAgent):
    """
    Standard greedy heuristic agent.
    A solid benchmark above RandomAgent.
    """

    def __init__(self, name: str = "Greedy"):
        super().__init__(name=name)

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        # 1. Capture any box if available (prioritize completing 2 boxes over 1)
        capturing_moves = []
        for m in valid_moves:
            captured_count = board.would_complete_box(m)
            if captured_count > 0:
                capturing_moves.append((captured_count, m))
                
        if capturing_moves:
            # Pick move that captures the maximum number of boxes (2 over 1)
            capturing_moves.sort(key=lambda x: x[0], reverse=True)
            best_capture_count = capturing_moves[0][0]
            best_captures = [m for count, m in capturing_moves if count == best_capture_count]
            return random.choice(best_captures)

        # 2. Look for "safe" moves (moves that do not give away a box by creating a 3rd edge)
        safe_moves = [m for m in valid_moves if not board.would_give_away_box(m)]
        if safe_moves:
            return random.choice(safe_moves)

        # 3. If forced to give away a box, pick an edge that gives away the fewest boxes
        # (e.g., an edge adjacent to only 1 box instead of 2 boxes)
        min_sacrifice = 999
        least_bad_moves = []
        for m in valid_moves:
            # Count how many boxes would become 3-sided
            count_3_sided = sum(1 for (r, c) in board.edge_to_boxes[m] if board.box_edge_counts[r][c] == 2)
            if count_3_sided < min_sacrifice:
                min_sacrifice = count_3_sided
                least_bad_moves = [m]
            elif count_3_sided == min_sacrifice:
                least_bad_moves.append(m)

        return random.choice(least_bad_moves)
