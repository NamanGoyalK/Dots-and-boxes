# Dots and Boxes: Comparative Study of AI Paradigms

A benchmarking platform for evaluating artificial intelligence approaches on the combinatorial game **Dots and Boxes**. The project implements and compares four distinct decision-making paradigms across arbitrary grid dimensions ($R \times C$).

---

## Implemented Approaches & Contributors

- **Rule-Based Heuristics (Ranjit)** (`agents/rule_based_agent.py`)  
  Applies Berlekamp's "strings and coins" combinatorial game theory. Prioritizes immediate box captures, filters out moves that yield 3-sided boxes, and implements the classic double-cross chain sacrifice strategy.

- **Deterministic Search: Minimax with Alpha-Beta Pruning (Shanmukh)** (`agents/minimax_agent.py`)  
  Adversarial tree search with branch pruning. Correctly models consecutive bonus turns (a player capturing a box retains their turn), uses zero-copy $O(1)$ state rollbacks via `undo_move()`, and applies move ordering to maximize cutoffs.

- **Probabilistic Search: Monte Carlo Tree Search (Saiyam)** (`agents/mcts_agent.py`)  
  Implements the standard four-phase MCTS lifecycle (Selection, Expansion, Simulation, Backpropagation). Balances exploitation and exploration via the UCB1 formula, with rollout policies evaluated over randomized playouts.

- **Reinforcement Learning: Q-Learning with Symmetry Reduction (Naman)** (`agents/q_learning_agent.py`)  
  Tabular Q-learning trained through self-play episodes. To address state-space explosion on larger grids, canonical states are computed across the 8 transformations of the dihedral group ($D_4$), collapsing state representations up to eight-fold.

---

## Project Structure

```
├── core/
│   ├── board.py            # Game state representation, bitmasks, D4 symmetries, O(1) undo
│   ├── base_agent.py       # Abstract BaseAgent interface
│   └── game.py             # Match orchestrator, move timing, and rule enforcement
├── agents/
│   ├── random_agent.py     # Uniform random baseline
│   ├── greedy_agent.py     # Greedy baseline
│   ├── human_agent.py      # Terminal input interface
│   ├── rule_based_agent.py # Rule-based expert (Ranjit)
│   ├── minimax_agent.py    # Alpha-Beta minimax (Shanmukh)
│   ├── mcts_agent.py       # Monte Carlo Tree Search (Saiyam)
│   └── q_learning_agent.py # Q-learning agent (Naman)
├── tournament/
│   ├── match.py            # Alternating-side multi-game series
│   └── benchmark.py        # Round-robin tournament engine
├── play.py                 # CLI interface for manual and automated matches
├── run_tournament.py       # Benchmark runner script
├── gui.py                  # Tkinter visualizer
└── tests/
    └── test_board.py       # Unit test suite
```

---

## Getting Started

### Prerequisites
Python 3.8+ with standard library and `numpy`.

### Running Matches (CLI)
```bash
# Play against the greedy baseline
python3 play.py --p1 human --p2 greedy

# Run a match between Minimax and MCTS
python3 play.py --p1 minimax --p2 mcts --delay 0.1
```

### Graphical Interface
```bash
python3 gui.py
```
Provides an interactive Tkinter board for human play and live AI duels.

### Running the Tournament Benchmark
```bash
# Full tournament (3x3 grid, 6 games per matchup, alternating first turn)
python3 run_tournament.py --rows 3 --cols 3 --games 6

# Fast smoke test (2x2 grid)
python3 run_tournament.py --quick
```

---

## Experimental Benchmark

Sample results from a round-robin tournament on a $3 \times 3$ grid (9 boxes, 24 edges), with 10 alternating games per matchup:

| Rank | Agent | Record (W-L-D) | Win Rate | Avg Points | Avg Move Latency |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Greedy Heuristic | 9-1-0 | 90.0% | 7.10 | 0.01 ms |
| 2 | Rule-Based Expert (Ranjit) | 8-2-0 | 80.0% | 7.00 | 0.01 ms |
| 3 | Alpha-Beta Minimax (Shanmukh) | 7-3-0 | 70.0% | 6.30 | 0.38 ms |
| 4 | Monte Carlo Tree Search (Saiyam) | 3-7-0 | 30.0% | 2.70 | 12.92 ms |
| 5 | Random Baseline | 2-8-0 | 20.0% | 2.10 | 0.00 ms |
| 6 | Q-Learning RL (Naman) | 1-9-0 | 10.0% | 1.80 | 0.05 ms |

---

## Tests

Execute the unit test suite:
```bash
python3 -m unittest discover tests -v
```
Verifies edge indexing, box completion logic, bonus turn handling, move retraction (`undo_move`), board cloning, and dihedral symmetry equivalences.
