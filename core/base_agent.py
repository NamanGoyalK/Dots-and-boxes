"""
Base Agent class for Dots and Boxes.
All AI and baseline agents inherit from BaseAgent.
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
from core.board import Board


class BaseAgent(ABC):
    """
    Abstract base class for all Dots and Boxes players.
    
    Subclasses must implement:
        get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]
        
    Moves can be returned as:
        - An integer edge ID: 0 <= edge_id < board.total_edges
        - A tuple: ('H', r, c) or ('V', r, c)
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.player_id: Optional[int] = None  # Will be set to 1 or 2 by Game controller

    @abstractmethod
    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        """
        Choose and return a move for the current board state.
        
        Args:
            board: Current game Board instance (read-only or clone for tree searches).
            
        Returns:
            The selected legal move as an integer ID or ('H'/'V', r, c) tuple.
        """
        pass

    def on_game_start(self, player_id: int, board: Board) -> None:
        """
        Hook called at the start of a match.
        Useful for resetting game-specific cache or initializing state.
        """
        self.player_id = player_id

    def on_game_end(self, winner: Optional[int], final_scores: dict, board: Board) -> None:
        """
        Hook called when a game concludes.
        Useful for Reinforcement Learning (Q-learning) to update weights or log metrics.
        """
        pass

    def reset(self) -> None:
        """Reset internal agent memory / cache between separate matches if needed."""
        pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
