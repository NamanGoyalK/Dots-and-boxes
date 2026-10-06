"""
Probabilistic Search Agent
Algorithm: Monte Carlo Tree Search (MCTS)

Author: Saiyam

Strategic Principles & Implementation Notes:
1. Four Phases of MCTS:
   - Selection: Traverse tree using UCB1 formula until reaching an unexpanded node.
   - Expansion: Expand a child node corresponding to an untried legal move.
   - Simulation (Rollout): Rapidly play random moves to game completion.
   - Backpropagation: Propagate win/loss result up the tree to update visit counts and scores.
2. UCB1 Formula:
   UCB1 = (w_i / n_i) + c * sqrt(ln(N) / n_i)
   - w_i: wins/score accumulated by child
   - n_i: number of visits to child
   - N: total visits to parent node
   - c: exploration constant (typically sqrt(2) ≈ 1.414)
3. Rollout Speed:
   - High-throughput simulation loops ensure robust tree convergence.
"""

import math
import random
import time
from typing import Union, Tuple, Optional, List, Dict
from core.base_agent import BaseAgent
from core.board import Board


class MCTSNode:
    """Represents a state node in the Monte Carlo search tree."""

    def __init__(
        self,
        board: Board,
        parent: Optional['MCTSNode'] = None,
        move_taken: Optional[int] = None,
        player_just_moved: Optional[int] = None
    ):
        self.board_state = board.clone()
        self.parent = parent
        self.move_taken = move_taken
        self.player_just_moved = player_just_moved
        
        self.children: Dict[int, 'MCTSNode'] = {}
        self.untried_moves = board.get_valid_moves()
        random.shuffle(self.untried_moves)
        
        self.visits: int = 0
        self.wins: float = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.board_state.is_game_over()

    def ucb1_score(self, child: 'MCTSNode', exploration_constant: float = 1.414) -> float:
        if child.visits == 0:
            return math.inf
        exploitation = child.wins / child.visits
        exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + exploration


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search Agent with UCB1 node selection.
    """

    def __init__(
        self,
        name: str = "Monte Carlo Tree Search (Saiyam)",
        iterations: int = 300,
        time_limit: Optional[float] = None,
        exploration_constant: float = 1.414
    ):
        super().__init__(name=name)
        self.iterations = iterations
        self.time_limit = time_limit
        self.c = exploration_constant

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")
        if len(valid_moves) == 1:
            return valid_moves[0]

        root = MCTSNode(
            board=board,
            parent=None,
            move_taken=None,
            player_just_moved=3 - board.current_player
        )

        start_time = time.perf_counter()
        sim_count = 0

        while True:
            if self.time_limit is not None:
                if (time.perf_counter() - start_time) >= self.time_limit:
                    break
            else:
                if sim_count >= self.iterations:
                    break

            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)

            # 3. Simulation (Rollout)
            winner = self._simulate(node.board_state)

            # 4. Backpropagation
            self._backpropagate(node, winner)

            sim_count += 1

        if not root.children:
            return random.choice(valid_moves)

        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB1 until reaching an unexpanded or terminal node."""
        while not node.is_terminal() and node.is_fully_expanded():
            best_score = -math.inf
            best_child = None
            for child in node.children.values():
                score = node.ucb1_score(child, self.c)
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Pick an untried move and add a new child node to the tree."""
        move = node.untried_moves.pop()
        next_board = node.board_state.clone()
        player_moving = next_board.current_player
        next_board.make_move(move)

        child = MCTSNode(
            board=next_board,
            parent=node,
            move_taken=move,
            player_just_moved=player_moving
        )
        node.children[move] = child
        return child

    def _simulate(self, board: Board) -> Optional[int]:
        """Fast rollout phase: play out game to terminal state."""
        sim_board = board.clone()
        while not sim_board.is_game_over():
            valid = sim_board.get_valid_moves()
            
            # Rollout policy: take box completion when immediately available
            captures = [m for m in valid if sim_board.would_complete_box(m) > 0]
            if captures:
                move = random.choice(captures)
            else:
                move = random.choice(valid)
                
            sim_board.make_move(move)
            
        return sim_board.get_winner()

    def _backpropagate(self, node: MCTSNode, winner: Optional[int]) -> None:
        """Propagate result up to root node."""
        curr = node
        while curr is not None:
            curr.visits += 1
            if winner == 0:
                curr.wins += 0.5
            elif winner == curr.player_just_moved:
                curr.wins += 1.0
            else:
                curr.wins += 0.0
            curr = curr.parent
