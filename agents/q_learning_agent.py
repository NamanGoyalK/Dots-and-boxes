"""
Reinforcement Learning Agent
Algorithm: Q-Learning with Canonical Dihedral Symmetry Reduction & Negamax Self-Play

Author: Naman

Architecture & Design:
1. State & Action Compression via Dihedral Symmetries (D4):
   - A raw 3x3 Dots and Boxes grid has 24 edges -> 2^24 ≈ 16.7 million states.
   - By mapping every state to its canonical minimal representative across the 8 dihedral symmetries,
     we compress the unique state space up to 8-fold.
   - Actions are transformed into the canonical coordinate frame:
     canonical_action = board._transform_edge(action, rot, flip)
     ensuring complete coordinate consistency in tabular Q-learning.
2. Two-Player Zero-Sum (Negamax) Self-Play:
   - In Dots and Boxes, capturing a box awards an immediate bonus turn.
   - When extra_turn is awarded:
     target = reward + gamma * max_a' Q(s', a')   (same player continues)
   - When turn passes to the opponent:
     target = reward - gamma * max_a' Q(s', a')   (negamax zero-sum value inversion)
3. Strategic Candidate Move Evaluation & Reward Shaping:
   - Capturing moves are evaluated first; safe moves second; minimal sacrifices last.
   - Learned Q-values select the optimal choice among candidate lines.
4. Model Persistence & Auto-Loading:
   - Supports saving and loading trained Q-tables.
   - Automatically loads pre-trained models for the board size if available.
"""

import os
import pickle
import random
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
from core.base_agent import BaseAgent
from core.board import Board


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent with dihedral symmetry reduction and tabular negamax self-play.
    """

    def __init__(
        self,
        name: str = "Q-Learning RL (Naman)",
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
        use_symmetries: bool = True,
        filter_candidates: bool = True,
        model_path: Optional[str] = None,
        auto_load: bool = True
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_symmetries = use_symmetries
        self.filter_candidates = filter_candidates
        self.auto_load = auto_load
        self.model_path = model_path
        
        # Q-table: canonical_state_key -> {canonical_action: q_value}
        self.q_table: Dict[int, Dict[int, float]] = {}
        self.training_mode: bool = False
        self.loaded_dims: Optional[Tuple[int, int]] = None

        if model_path:
            self.load_model(model_path)

    # --- Symmetry & State Helpers ---

    def _get_state_and_trans(self, board: Board) -> Tuple[int, Tuple[int, bool]]:
        """
        Returns (state_key, (rot, flip)).
        If symmetries are enabled and board is square, state_key is the canonical minimal mask.
        """
        if self.use_symmetries and board.rows == board.cols:
            return board.get_canonical_transformation()
        return board.get_state_key()[0], (0, False)

    def _get_state(self, board: Board) -> Tuple[int, int]:
        """
        Backward-compatible helper returning (canonical_mask, current_player).
        """
        state_key, _ = self._get_state_and_trans(board)
        return (state_key, board.current_player)

    def _map_action_to_canonical(self, board: Board, action: int, trans: Tuple[int, bool]) -> int:
        """Map raw board action to canonical orientation."""
        if self.use_symmetries and board.rows == board.cols:
            return board._transform_edge(action, trans[0], trans[1])
        return action

    # --- Candidate Moves & Strategy ---

    def get_candidate_moves(self, board: Board, valid_moves: Optional[List[int]] = None) -> List[int]:
        """
        Prunes obvious tactical blunders to focus policy search on strategic lines:
        1. Captures: Completing 3-sided boxes immediately.
        2. Safe moves: Moves that do not create 3-sided boxes (giving away boxes).
        3. Forced sacrifices: Minimal damage (sacrificing smallest chain/box).
        """
        if valid_moves is None:
            valid_moves = board.get_valid_moves()
            
        if not self.filter_candidates:
            return valid_moves

        captures = [m for m in valid_moves if board.would_complete_box(m) > 0]
        if captures:
            return captures

        safes = [m for m in valid_moves if not board.would_give_away_box(m)]
        if safes:
            return safes

        # Forced sacrifice: find moves creating minimum 3-sided boxes
        min_damage = 999
        least_bad = []
        for m in valid_moves:
            dmg = sum(1 for (r, c) in board.edge_to_boxes[m] if board.box_edge_counts[r][c] == 2)
            if dmg < min_damage:
                min_damage = dmg
                least_bad = [m]
            elif dmg == min_damage:
                least_bad.append(m)
        return least_bad if least_bad else valid_moves

    # --- Q-Value Accessors ---

    def get_q_value(self, state_key: int, canonical_action: int) -> float:
        """Retrieve Q-value for a canonical state-action pair."""
        return self.q_table.get(state_key, {}).get(canonical_action, 0.0)

    def set_q_value(self, state_key: int, canonical_action: int, value: float) -> None:
        """Store Q-value for a canonical state-action pair."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][canonical_action] = value

    def get_q_values(self, state_key: Union[int, Tuple], valid_moves: List[int]) -> Dict[int, float]:
        """Backward-compatible helper."""
        key = state_key[0] if isinstance(state_key, tuple) else state_key
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in valid_moves}
        for a in valid_moves:
            if a not in self.q_table[key]:
                self.q_table[key][a] = 0.0
        return self.q_table[key]

    # --- Match & Move Decision ---

    def on_game_start(self, player_id: int, board: Board) -> None:
        super().on_game_start(player_id, board)
        if self.auto_load and (not self.q_table or self.loaded_dims != (board.rows, board.cols)):
            self._try_auto_load(board.rows, board.cols)

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")
        if len(valid_moves) == 1:
            return valid_moves[0]

        if self.auto_load and not self.q_table:
            self._try_auto_load(board.rows, board.cols)

        candidates = self.get_candidate_moves(board, valid_moves)
        if len(candidates) == 1:
            return candidates[0]

        # Epsilon-greedy exploration during training
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(candidates)

        state_key, trans = self._get_state_and_trans(board)
        q_dict = self.q_table.get(state_key, {})

        scored_moves = []
        for a in candidates:
            can_a = self._map_action_to_canonical(board, a, trans)
            q_val = q_dict.get(can_a, 0.0)
            scored_moves.append((q_val, a))

        max_q = max(val for val, _ in scored_moves)
        best_actions = [a for val, a in scored_moves if val == max_q]
        return random.choice(best_actions)

    # --- Q-Learning Bellman Update ---

    def update_q(
        self,
        state_key: Union[int, Tuple],
        action: int,
        reward: float,
        next_board: Board,
        extra_turn: bool = False,
        done: bool = False
    ) -> None:
        """
        Negamax two-player zero-sum Bellman update.
        Handles both signature variants for backward compatibility.
        """
        key = state_key[0] if isinstance(state_key, tuple) else state_key
        curr_q = self.get_q_value(key, action)

        if done:
            target = reward
        else:
            next_valid = next_board.get_valid_moves()
            if not next_valid:
                target = reward
            else:
                next_candidates = self.get_candidate_moves(next_board, next_valid)
                next_state_key, next_trans = self._get_state_and_trans(next_board)
                next_q_dict = self.q_table.get(next_state_key, {})
                
                max_next_q = -float('inf')
                for m in next_candidates:
                    can_m = self._map_action_to_canonical(next_board, m, next_trans)
                    q_val = next_q_dict.get(can_m, 0.0)
                    if q_val > max_next_q:
                        max_next_q = q_val
                        
                if max_next_q == -float('inf'):
                    max_next_q = 0.0

                if extra_turn:
                    # Same player continues
                    target = reward + self.gamma * max_next_q
                else:
                    # Turn passed to opponent (negamax value inversion)
                    target = reward - self.gamma * max_next_q

        new_q = curr_q + self.alpha * (target - curr_q)
        self.set_q_value(key, action, new_q)

    # --- Training Loop ---

    def train_self_play(
        self,
        num_episodes: int = 15000,
        rows: int = 2,
        cols: int = 2,
        verbose_interval: int = 2000
    ) -> None:
        """
        Train the agent via self-play with symmetry reduction and negamax updates.
        """
        self.training_mode = True
        self.loaded_dims = (rows, cols)
        print(f"Starting Q-Learning self-play training: {num_episodes} episodes on {rows}x{cols} board...")

        for episode in range(1, num_episodes + 1):
            board = Board(rows=rows, cols=cols)

            while not board.is_game_over():
                curr_player = board.current_player
                state_key, trans = self._get_state_and_trans(board)
                valid_moves = board.get_valid_moves()
                candidates = self.get_candidate_moves(board, valid_moves)

                # Epsilon-greedy move selection
                if random.random() < self.epsilon:
                    action = random.choice(candidates)
                else:
                    q_dict = self.q_table.get(state_key, {})
                    scored = []
                    for m in candidates:
                        can_m = self._map_action_to_canonical(board, m, trans)
                        scored.append((q_dict.get(can_m, 0.0), m))
                    max_q = max(v for v, _ in scored)
                    best_candidates = [m for v, m in scored if v == max_q]
                    action = random.choice(best_candidates)

                can_action = self._map_action_to_canonical(board, action, trans)

                # Execute move
                captured, extra_turn = board.make_move(action)
                done = board.is_game_over()

                # Step reward
                reward = float(captured * 2.0)

                # Terminal reward
                if done:
                    winner = board.get_winner()
                    if winner == curr_player:
                        terminal_reward = 10.0
                    elif winner == (3 - curr_player):
                        terminal_reward = -10.0
                    else:
                        terminal_reward = 0.0
                    total_reward = reward + terminal_reward
                else:
                    total_reward = reward

                # Negamax Bellman update
                self.update_q(
                    state_key=state_key,
                    action=can_action,
                    reward=total_reward,
                    next_board=board,
                    extra_turn=extra_turn,
                    done=done
                )

            # Anneal epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % verbose_interval == 0 or episode == num_episodes:
                print(
                    f"Episode {episode}/{num_episodes} | "
                    f"States in Q-table: {len(self.q_table)} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        self.training_mode = False
        print(f"Training complete! Final Q-table states: {len(self.q_table)}\n")

    # --- Persistence & Auto-Loading ---

    def _try_auto_load(self, rows: int, cols: int) -> bool:
        """Attempt to find and load a saved model for the given dimensions."""
        project_root = Path(__file__).resolve().parent.parent
        candidates = [
            project_root / "models" / f"q_table_{rows}x{cols}.pkl",
            project_root / f"q_table_{rows}x{cols}.pkl",
            project_root / "models" / "q_table.pkl",
            project_root / "q_table.pkl"
        ]

        for p in candidates:
            if p.is_file():
                try:
                    self.load_model(str(p))
                    self.loaded_dims = (rows, cols)
                    return True
                except Exception as e:
                    print(f"Warning: Failed to load model from {p}: {e}")
        return False

    def save_model(self, filepath: str) -> None:
        """Persist learned Q-table and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "q_table": self.q_table,
            "loaded_dims": self.loaded_dims,
            "use_symmetries": self.use_symmetries
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath} ({len(self.q_table)} states)")

    def load_model(self, filepath: str) -> None:
        """Load learned Q-table from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and "q_table" in data:
            self.q_table = data["q_table"]
            self.loaded_dims = data.get("loaded_dims")
        elif isinstance(data, dict):
            self.q_table = data
        else:
            raise ValueError(f"Unrecognized model data format in {filepath}")
            
        print(f"Model loaded from {filepath} ({len(self.q_table)} states)")
