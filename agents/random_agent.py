"""
Random Agent - Plays uniformly random legal moves.
Serves as the baseline control for benchmarking.
"""

import random
from typing import Union, Tuple
from core.base_agent import BaseAgent
from core.board import Board


class RandomAgent(BaseAgent):
    """Selects a uniformly random move among available legal edges."""

    def __init__(self, name: str = "Random"):
        super().__init__(name=name)

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")
        return random.choice(valid_moves)
