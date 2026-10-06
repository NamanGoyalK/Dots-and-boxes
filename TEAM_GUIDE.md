# 👥 Dots and Boxes AI Platform - Technical Development Guide

This guide outlines the unified interface, architecture, and developer workflows for each agent in our multi-agent Dots and Boxes platform.

---

## 🛠️ Unified Agent Interface (`BaseAgent`)

Every agent inherits from `core.base_agent.BaseAgent` and implements `get_move(board: Board)`:

```python
from core.base_agent import BaseAgent
from core.board import Board

class CustomAgent(BaseAgent):
    def __init__(self, name="Custom Agent"):
        super().__init__(name=name)

    def get_move(self, board: Board):
        """
        Choose and return a move for the current board state.
        Returns:
            - An integer edge_id in [0, board.total_edges - 1]
            - OR a tuple: ('H', r, c) / ('V', r, c)
        """
        valid_moves = board.get_valid_moves()
        return valid_moves[0]
```

### Essential Board Methods & Properties
| Method / Property | Description |
| :--- | :--- |
| `board.get_valid_moves()` | Returns list of remaining legal edge IDs. |
| `board.would_complete_box(edge)` | Returns number of boxes (0, 1, or 2) that would be captured by this edge. |
| `board.would_give_away_box(edge)` | Returns `True` if playing this edge leaves any adjacent box with 3 sides (giving it away). |
| `board.get_capturable_moves()` | Returns list of moves that immediately capture at least 1 box. |
| `board.get_safe_moves()` | Returns moves that either capture a box or do not create a 3rd side. |
| `board.make_move(edge)` | Executes move on the board; returns `(num_captured, extra_turn)`. |
| `board.undo_move()` | Reverts the last move in $O(1)$ time, restoring box ownership, scores, and turn. |
| `board.clone()` | Returns an independent deep copy of the board for tree rollouts. |
| `board.id_to_edge(edge_id)` | Converts integer edge ID to coordinate tuple `('H'/'V', r, c)`. |
| `board.edge_to_id(etype, r, c)` | Converts coordinate tuple `('H'/'V', r, c)` to flat edge ID. |

---

## 📌 Module 1: Rule-Based Expert Agent
- **Target File**: `agents/rule_based_agent.py`
- **Class**: `RuleBasedAgent`
- **Lead Developer**: Ranjit
- **Key Strategy & Tasks**:
  1. **Immediate Captures**:
     - Evaluate `board.get_capturable_moves()`. Prioritize capturing 2 boxes simultaneously over single-box captures.
  2. **Safe Edge Placement**:
     - Filter unplayed edges using `board.get_safe_moves()`.
     - Prefer moves that create 1-sided boxes over 2-sided boxes.
  3. **The Double-Cross Rule (Berlekamp Theory)**:
     - When capturing a corridor/loop, offering the last 2 boxes to the opponent forces them to open the subsequent chain.
- **Testing Command**:
  ```bash
  python3 play.py --p1 rule_based --p2 greedy --delay 0.1
  ```

---

## 📌 Module 2: Deterministic Search (Minimax + Alpha-Beta)
- **Target File**: `agents/minimax_agent.py`
- **Class**: `MinimaxAgent`
- **Lead Developer**: Shanmukh
- **Key Strategy & Tasks**:
  1. **Consecutive Turn Handling**:
     - In Dots and Boxes, capturing a box awards an immediate extra turn.
     - When a capture occurs during tree search, the recursive step evaluates the **same** player.
  2. **Zero-Copy Tree Traversal**:
     - Avoid creating board copies at search nodes.
     - Use `captured, extra = board.make_move(move)`, recurse, and then backtrack using `board.undo_move()`.
  3. **Move Ordering for Alpha-Beta Cutoffs**:
     - Evaluate capturing moves first, safe moves second, and sacrifices last to prune branches early.
  4. **Heuristic Evaluation**:
     - Refine leaf node heuristic scoring using score differentials and chain potential.
- **Testing Command**:
  ```bash
  python3 play.py --p1 minimax --p2 rule_based --delay 0.1
  ```

---

## 📌 Module 3: Probabilistic Search (MCTS)
- **Target File**: `agents/mcts_agent.py`
- **Class**: `MCTSAgent`
- **Lead Developer**: Saiyam
- **Key Strategy & Tasks**:
  1. **MCTS Phases**:
     - **Selection**: Traverse the tree using UCB1: $\frac{w_i}{n_i} + c \sqrt{\frac{\ln N_i}{n_i}}$.
     - **Expansion**: Instantiate a child node for an untried legal move.
     - **Simulation (Rollout)**: Play out game transitions rapidly to terminal state.
     - **Backpropagation**: Update visit counts and victory credits up to the root.
  2. **Rollout Policy Optimization**:
     - Incorporate light heuristic moves (e.g. taking immediate box captures) during rollout to simulate realistic play.
  3. **Hyperparameter Tuning**:
     - Tune exploration constant $c$ (typically $1.0 \le c \le 2.0$) and iteration budgets.
- **Testing Command**:
  ```bash
  python3 play.py --p1 mcts --p2 minimax --delay 0.1
  ```

---

## 📌 Module 4: Reinforcement Learning (Q-Learning)
- **Target File**: `agents/q_learning_agent.py`
- **Class**: `QLearningAgent`
- **Lead Developer**: Naman
- **Key Strategy & Tasks**:
  1. **Dihedral Symmetry Compression ($D_4$)**:
     - A 3x3 board contains $2^{24} \approx 16.7\text{M}$ states.
     - Use `board.get_canonical_state()` to map every board configuration into its minimal representation across 8 rotations and reflections, compressing tabular storage up to 8-fold.
  2. **Self-Play Training Loop**:
     - Run `agent.train_self_play(num_episodes=5000, rows=2, cols=2)`.
     - Persist learned state-action values via `agent.save_model("q_table.pkl")` and `agent.load_model("q_table.pkl")`.
- **Testing Command**:
  ```bash
  python3 play.py --p1 qlearning --p2 greedy --delay 0.1
  ```

---

## 🏁 Running the Tournament & Benchmarking
Run the round-robin tournament across all agents:
```bash
python3 run_tournament.py --games 6 --rows 3 --cols 3
```
This generates:
- Head-to-head win-loss-draw matrix
- Comprehensive leaderboard with Win Rate (%), Average Points, and Decision Latency (ms)
- Exported JSON data in `tournament_results.json`
