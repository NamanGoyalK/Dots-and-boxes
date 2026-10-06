"""
Deterministic Search Agent
Algorithm: Minimax with Alpha-Beta Pruning

Author: Shanmukh

Strategic Principles & Implementation Notes:
1. Alpha-Beta Pruning:
   - alpha: best score guaranteed to maximizing player so far.
   - beta: best score guaranteed to minimizing player so far.
   - Pruning cutoff occurs when beta <= alpha.
2. Bonus Turn Handling:
   - In Dots and Boxes, capturing a box awards an immediate extra turn.
   - In the minimax game tree, when a player completes a box, the recursive child
     remains for the same player.
3. Efficient Backtracking via undo_move:
   - Board state transitions are inverted using board.undo_move(), eliminating memory cloning.
4. Move Ordering:
   - Capturing moves are evaluated first, safe moves second, and sacrifices last,
     maximizing alpha-beta branch pruning efficiency.
"""

import math
from typing import Union, Tuple, List
from core.base_agent import BaseAgent
from core.board import Board


class MinimaxAgent(BaseAgent):
    """
    Adversarial search agent using Minimax with Alpha-Beta pruning.
    """

    def __init__(self, name: str = "Alpha-Beta Minimax (Shanmukh)", max_depth: int = 4):
        super().__init__(name=name)
        self.max_depth = max_depth
        self.nodes_evaluated = 0

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        if len(valid_moves) == 1:
            return valid_moves[0]

        self.nodes_evaluated = 0
        my_player = board.current_player
        
        best_move = valid_moves[0]
        alpha = -math.inf
        beta = math.inf
        best_val = -math.inf

        ordered_moves = self._order_moves(board, valid_moves)

        for move in ordered_moves:
            captured, extra_turn = board.make_move(move)
            
            if board.is_game_over():
                score = self._evaluate_terminal(board, my_player)
            elif extra_turn:
                # Same player continues moving
                score = self._minimax(board, self.max_depth - 1, alpha, beta, is_my_turn=True, my_player=my_player)
            else:
                # Turn passed to opponent
                score = self._minimax(board, self.max_depth - 1, alpha, beta, is_my_turn=False, my_player=my_player)
                
            board.undo_move()
            
            if score > best_val:
                best_val = score
                best_move = move
                
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break

        return best_move

    def _minimax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        is_my_turn: bool,
        my_player: int
    ) -> float:
        self.nodes_evaluated += 1

        if board.is_game_over():
            return self._evaluate_terminal(board, my_player)

        if depth <= 0:
            return self._heuristic_eval(board, my_player)

        valid_moves = self._order_moves(board, board.get_valid_moves())

        if is_my_turn:
            max_eval = -math.inf
            for move in valid_moves:
                captured, extra_turn = board.make_move(move)
                
                if board.is_game_over():
                    val = self._evaluate_terminal(board, my_player)
                elif extra_turn:
                    val = self._minimax(board, depth - 1, alpha, beta, is_my_turn=True, my_player=my_player)
                else:
                    val = self._minimax(board, depth - 1, alpha, beta, is_my_turn=False, my_player=my_player)
                    
                board.undo_move()
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in valid_moves:
                captured, extra_turn = board.make_move(move)
                
                if board.is_game_over():
                    val = self._evaluate_terminal(board, my_player)
                elif extra_turn:
                    val = self._minimax(board, depth - 1, alpha, beta, is_my_turn=False, my_player=my_player)
                else:
                    val = self._minimax(board, depth - 1, alpha, beta, is_my_turn=True, my_player=my_player)
                    
                board.undo_move()
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_terminal(self, board: Board, my_player: int) -> float:
        """Terminal game payoff."""
        opp_player = 3 - my_player
        my_score = board.scores[my_player]
        opp_score = board.scores[opp_player]
        
        if my_score > opp_score:
            return 1000.0 + (my_score - opp_score) * 10.0
        elif opp_score > my_score:
            return -1000.0 - (opp_score - my_score) * 10.0
        else:
            return 0.0

    def _heuristic_eval(self, board: Board, my_player: int) -> float:
        """
        Evaluation function when max search depth is reached.
        """
        opp_player = 3 - my_player
        score_diff = board.scores[my_player] - board.scores[opp_player]
        
        three_sided = 0
        for r in range(board.rows):
            for c in range(board.cols):
                if board.box_edge_counts[r][c] == 3:
                    three_sided += 1
                    
        turn_mult = 1.0 if board.current_player == my_player else -1.0
        return (score_diff * 10.0) + (three_sided * 2.0 * turn_mult)

    def _order_moves(self, board: Board, moves: List[int]) -> List[int]:
        """Move ordering to maximize Alpha-Beta pruning cutoffs."""
        captures = []
        safe = []
        sacrifices = []
        
        for m in moves:
            if board.would_complete_box(m) > 0:
                captures.append(m)
            elif not board.would_give_away_box(m):
                safe.append(m)
            else:
                sacrifices.append(m)
                
        return captures + safe + sacrifices
