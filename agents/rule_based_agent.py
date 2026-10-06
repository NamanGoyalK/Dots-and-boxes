"""
Rule-Based Expert Agent
Algorithm: Advanced Greedy & "Strings and Coins" Logic

Author: Ranjit

Strategic Principles:
1. Priority 1 (Box Capture):
   - If any box has 3 lines drawn, capture it immediately.
   - Prioritize double-box captures (completing 2 boxes at once) over single captures.
2. Priority 2 (Safe Moves):
   - Avoid drawing the 3rd line of any box (which surrenders it to the opponent).
   - Prefer moves that open 1-sided boxes over 2-sided boxes.
3. Priority 3 (The Double-Cross Strategy):
   - In Dots and Boxes endgame theory (Berlekamp's "The Dots and Boxes Game"),
     a player capturing a long chain deliberately sacrifices the last 2 boxes
     (the "hard-hearted handout") by drawing an edge that leaves them to the opponent.
   - Whoever takes the last 2 boxes is forced to make the next move, opening the next chain.
4. Graph Theory ("Strings and Coins"):
   - Each box is a coin; each unplayed edge is a string.
   - Control of long chains determines match victory.
"""

import random
from typing import Union, Tuple, List, Optional
from core.base_agent import BaseAgent
from core.board import Board


class RuleBasedAgent(BaseAgent):
    """
    Expert Agent: Advanced Rule-Based & 'Strings and Coins' logic.
    """

    def __init__(self, name: str = "Rule-Based Expert (Ranjit)", enable_double_cross: bool = True):
        super().__init__(name=name)
        self.enable_double_cross = enable_double_cross

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        # ====================================================================
        # Priority 1: Capturing & Double-Cross Strategy
        # ====================================================================
        capturable = board.get_capturable_moves()
        if capturable:
            if self.enable_double_cross:
                dc_move = self._check_double_cross(board, capturable)
                if dc_move is not None:
                    return dc_move
            
            # Prioritize double-box captures (completing 2 boxes)
            capturable_with_scores = [(board.would_complete_box(m), m) for m in capturable]
            capturable_with_scores.sort(key=lambda x: x[0], reverse=True)
            max_score = capturable_with_scores[0][0]
            best_moves = [m for score, m in capturable_with_scores if score == max_score]
            return random.choice(best_moves)

        # ====================================================================
        # Priority 2: Safe Moves (Avoid creating a 3rd line on any box)
        # ====================================================================
        safe_moves = board.get_safe_moves()
        if safe_moves:
            return self._select_best_safe_move(board, safe_moves)

        # ====================================================================
        # Priority 3: Minimal Sacrifice
        # When forced to sacrifice, choose the move giving away the fewest boxes.
        # ====================================================================
        return self._select_minimal_sacrifice_move(board, valid_moves)

    def _check_double_cross(self, board: Board, capturable_moves: List[int]) -> Optional[int]:
        """
        Implements the 'Double-Cross' strategy from Dots and Boxes theory.
        If we are in a chain of boxes and only 2 boxes remain in this chain,
        instead of capturing both and being forced to open the next chain,
        we place a line that offers those 2 boxes to the opponent.
        """
        # Ranjit: Implement corridor/chain length detection here
        return None

    def _select_best_safe_move(self, board: Board, safe_moves: List[int]) -> int:
        """
        Among safe moves, prefer moves that maintain long-term safety.
        Creating a 1-sided box is safer than creating a 2-sided box.
        """
        scored_moves = []
        for m in safe_moves:
            score = 0
            for (r, c) in board.edge_to_boxes[m]:
                edges = board.box_edge_counts[r][c]
                if edges == 0:
                    score += 2  # Very safe: box now has 1 edge
                elif edges == 1:
                    score += 1  # Moderate: box now has 2 edges
            scored_moves.append((score, m))
            
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        best_score = scored_moves[0][0]
        candidates = [m for s, m in scored_moves if s == best_score]
        return random.choice(candidates)

    def _select_minimal_sacrifice_move(self, board: Board, valid_moves: List[int]) -> int:
        """
        When forced to sacrifice, surrender the smallest possible chain.
        """
        min_damage = 999
        best_moves = []
        for m in valid_moves:
            damage = sum(1 for (r, c) in board.edge_to_boxes[m] if board.box_edge_counts[r][c] == 2)
            if damage < min_damage:
                min_damage = damage
                best_moves = [m]
            elif damage == min_damage:
                best_moves.append(m)
        return random.choice(best_moves)
