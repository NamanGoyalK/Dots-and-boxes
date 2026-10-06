"""
Deep Q-Network (DQN) Agent for Dots and Boxes.
Supports GPU acceleration (CUDA) on NVIDIA GPUs (e.g., RTX 4060) via PyTorch.

Ideal for larger boards (such as 4x4 with 2^40 ≈ 1.1 trillion states)
where tabular state spaces exceed available RAM.
"""

import os
import random
from collections import deque
from typing import Union, Tuple, List, Optional
from core.base_agent import BaseAgent
from core.board import Board

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """
        Deep Q-Network for approximating Q(s, a).
        Maps flat board edge feature vector to Q-values for every edge.
        """
        def __init__(self, num_edges: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(num_edges + 1, hidden_dim),  # edges + current_player indicator
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_edges)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent leveraging PyTorch and NVIDIA CUDA acceleration.
    """

    def __init__(
        self,
        name: str = "Deep Q-Network (GPU/CUDA)",
        rows: int = 4,
        cols: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 0.05,
        device: Optional[str] = None
    ):
        super().__init__(name=name)
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.epsilon = epsilon
        self.training_mode = False

        if not TORCH_AVAILABLE:
            self.device = "cpu"
            self.q_net = None
            return

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        temp_board = Board(rows=rows, cols=cols)
        self.num_edges = temp_board.total_edges
        
        self.q_net = QNetwork(self.num_edges).to(self.device)
        self.target_net = QNetwork(self.num_edges).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=20000)

    def _board_to_tensor(self, board: Board) -> torch.Tensor:
        """Convert board state into a tensor representation."""
        edge_features = [1.0 if e else 0.0 for e in board.edges]
        features = edge_features + [float(board.current_player)]
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available.")
        if len(valid_moves) == 1:
            return valid_moves[0]

        # Prioritize immediate captures and safe moves
        captures = [m for m in valid_moves if board.would_complete_box(m) > 0]
        if captures:
            return random.choice(captures)

        if not TORCH_AVAILABLE or self.q_net is None:
            # Safe heuristic fallback if PyTorch is not installed
            safes = [m for m in valid_moves if not board.would_give_away_box(m)]
            return random.choice(safes if safes else valid_moves)

        if self.training_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)

        self.q_net.eval()
        with torch.no_grad():
            state_t = self._board_to_tensor(board)
            q_values = self.q_net(state_t).squeeze(0)
            
            # Mask illegal moves with -inf
            masked_q = {m: q_values[m].item() for m in valid_moves}
            best_move = max(masked_q.items(), key=lambda x: x[1])[0]
            return best_move
