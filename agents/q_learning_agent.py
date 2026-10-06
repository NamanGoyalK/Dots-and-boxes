"""
Reinforcement Learning Agent
Algorithm: Q-Learning with Canonical Dihedral Symmetry Reduction

Author: Naman

Architecture & Design:
1. State Compression via Symmetries:
   - A raw 3x3 Dots and Boxes grid has 24 edges -> 2^24 ≈ 16.7 million states.
   - By mapping every state to its canonical representative across the 8 dihedral symmetries (D4),
     we compress the unique state space up to 8-fold.
2. Self-Play Reinforcement Learning:
   - Agent trains by playing against copies of itself.
   - Bellman equation update:
     Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
3. Reward Shaping:
   - Immediate positive reward for capturing boxes.
   - Terminal reward for winning the match.
4. Epsilon-Greedy Exploration:
   - Anneals from epsilon_start to epsilon_min over training episodes.
"""

import pickle
import random
from typing import Union, Tuple, Dict, List
from core.base_agent import BaseAgent
from core.board import Board


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent with symmetry reduction and tabular self-play.
    """

    def __init__(
        self,
        name: str = "Q-Learning RL (Naman)",
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
        use_symmetries: bool = True
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_symmetries = use_symmetries
        
        # Q-table: state_key -> {action: q_value}
        self.q_table: Dict[Tuple, Dict[int, float]] = {}
        self.training_mode = False

    def _get_state(self, board: Board) -> Tuple:
        if self.use_symmetries:
            return board.get_canonical_state()
        return board.get_state_key()

    def get_q_values(self, state_key: Tuple, valid_moves: List[int]) -> Dict[int, float]:
        """Retrieve or initialize Q-values for valid actions in the state."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in valid_moves}
        else:
            for a in valid_moves:
                if a not in self.q_table[state_key]:
                    self.q_table[state_key][a] = 0.0
        return self.q_table[state_key]

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")

        state_key = self._get_state(board)
        q_vals = self.get_q_values(state_key, valid_moves)

        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)

        max_q = max(q_vals[a] for a in valid_moves)
        best_actions = [a for a in valid_moves if q_vals[a] == max_q]
        return random.choice(best_actions)

    def update_q(
        self,
        state_key: Tuple,
        action: int,
        reward: float,
        next_board: Board,
        done: bool
    ) -> None:
        """Standard Bellman equation Q-learning update."""
        valid_next_moves = next_board.get_valid_moves()
        current_q = self.q_table[state_key].get(action, 0.0)

        if done or not valid_next_moves:
            target = reward
        else:
            next_state_key = self._get_state(next_board)
            next_q_vals = self.get_q_values(next_state_key, valid_next_moves)
            max_next_q = max(next_q_vals[a] for a in valid_next_moves)
            target = reward + self.gamma * max_next_q

        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)

    def train_self_play(
        self,
        num_episodes: int = 5000,
        rows: int = 2,
        cols: int = 2,
        verbose_interval: int = 1000
    ) -> None:
        """Train the agent via self-play episodes."""
        self.training_mode = True
        print(f"Starting Q-Learning self-play training: {num_episodes} episodes on {rows}x{cols} board...")

        for episode in range(1, num_episodes + 1):
            board = Board(rows=rows, cols=cols)
            step_history = []

            while not board.is_game_over():
                curr_player = board.current_player
                state_key = self._get_state(board)
                action = self.get_move(board)

                captured, extra_turn = board.make_move(action)
                step_reward = captured * 2.0
                step_history.append((curr_player, state_key, action, step_reward, board.clone(), board.is_game_over()))

            winner = board.get_winner()
            for (player, state, action, step_rew, next_b, done) in step_history:
                if winner == 0:
                    terminal_reward = 0.0
                elif winner == player:
                    terminal_reward = 5.0
                else:
                    terminal_reward = -5.0
                    
                total_reward = step_rew + (terminal_reward if done else 0.0)
                self.update_q(state, action, total_reward, next_b, done)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % verbose_interval == 0:
                print(f"Episode {episode}/{num_episodes} | States in Q-table: {len(self.q_table)} | Epsilon: {self.epsilon:.4f}")

        self.training_mode = False
        print(f"Training complete! Final Q-table states: {len(self.q_table)}\n")

    def save_model(self, filepath: str) -> None:
        """Persist learned Q-table to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filepath} ({len(self.q_table)} states)")

    def load_model(self, filepath: str) -> None:
        """Load learned Q-table from disk."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {filepath} ({len(self.q_table)} states)")
