# Dots and Boxes - Multi-Agent AI Arena & Benchmarking Platform

A high-performance, modular Python platform for developing, analyzing, and benchmarking artificial intelligence agents on the classic combinatorial game **Dots and Boxes**.

---

## 🎯 AI Architectures & Team Contributions

The platform incorporates four distinct algorithmic paradigms across artificial intelligence and game theory:

1. **Rule-Based Expert Agent** (`agents/rule_based_agent.py`)
   - **Author**: Ranjit
   - **Paradigm**: Advanced Greedy & "Strings and Coins" Combinatorial Graph Logic
   - **Techniques**: Immediate capture prioritization, safe edge isolation, and the Berlekamp Double-Cross chain sacrifice strategy.

2. **Alpha-Beta Minimax Agent** (`agents/minimax_agent.py`)
   - **Author**: Shanmukh
   - **Paradigm**: Deterministic Adversarial Game Tree Search
   - **Techniques**: Recursive Minimax with Alpha-Beta branch pruning, specialized handling of Dots & Boxes bonus turns, $O(1)$ state backtracking via `undo_move()`, and dynamic heuristic evaluation.

3. **Monte Carlo Tree Search Agent** (`agents/mcts_agent.py`)
   - **Author**: Saiyam
   - **Paradigm**: Probabilistic Search & Dynamic Sampling
   - **Techniques**: 4-phase MCTS lifecycle (Selection, Expansion, Simulation, Backpropagation), Upper Confidence Bound for Trees (UCB1) exploration/exploitation balance, and high-throughput rollout policies.

4. **Q-Learning Reinforcement Learning Agent** (`agents/q_learning_agent.py`)
   - **Author**: Naman
   - **Paradigm**: Model-Free Reinforcement Learning
   - **Techniques**: Tabular Q-Learning via self-play episodes, Bellman optimality equation updates, and state space reduction using canonical transformations under the 8 dihedral symmetries ($D_4$).

---

## ⚡ Quickstart

### 1. Interactive CLI Match
Play directly in the terminal (Human vs AI, AI vs AI):
```bash
# Play as Human against Greedy Heuristic
python3 play.py --p1 human --p2 greedy

# Watch Minimax face MCTS in terminal
python3 play.py --p1 minimax --p2 mcts --delay 0.2
```

### 2. Graphical Arena (Tkinter)
Launch the interactive desktop interface:
```bash
python3 gui.py
```
*Point-and-click to place edges, select any AI opponent, or watch live AI vs AI matches in real time.*

### 3. Run Benchmark Tournament
Execute a round-robin tournament across all agents with metrics (Win Rate, Avg Points, Move Latency):
```bash
# Full tournament on standard 3x3 board (24 edges, 9 boxes)
python3 run_tournament.py --rows 3 --cols 3 --games 6

# Quick smoke test on 2x2 board
python3 run_tournament.py --quick
```

---

## 📐 Board Representation & Coordinate System

An $R \times C$ board consists of:
- **Horizontal Edges**: $(R + 1) \times C$ edges, identified by `('H', r, c)`
- **Vertical Edges**: $R \times (C + 1)$ edges, identified by `('V', r, c)`
- **Total Edges**: $2RC + R + C$ (24 edges for $3 \times 3$, 12 edges for $2 \times 2$)

### Edge Indexing
All agents can return **either**:
1. An integer `edge_id` $\in [0, \text{total\_edges} - 1]$, OR
2. A coordinate tuple: `('H', r, c)` or `('V', r, c)`.

```
2x2 Board Edge Layout:
Horizontal Edges:
  H(0,0)=0,  H(0,1)=1
  H(1,0)=2,  H(1,1)=3
  H(2,0)=4,  H(2,1)=5
Vertical Edges:
  V(0,0)=6,  V(0,1)=7,  V(0,2)=8
  V(1,0)=9,  V(1,1)=10, V(1,2)=11
```

---

## 🏗️ Repository Architecture

```
Dots and boxes/
├── core/
│   ├── board.py            # Core engine: O(1) undo, bitmasking, D4 symmetries, ASCII renderer
│   ├── base_agent.py       # BaseAgent abstract class & lifecycle hooks
│   └── game.py             # Match orchestrator, timing profiler, rule validator, MatchResult
├── agents/
│   ├── random_agent.py     # Random baseline agent
│   ├── greedy_agent.py     # Heuristic greedy baseline
│   ├── human_agent.py      # Interactive console player
│   ├── rule_based_agent.py # Ranjit: Strings & coins logic, double-cross rule
│   ├── minimax_agent.py    # Shanmukh: Alpha-Beta pruning, bonus turn recursion, undo_move
│   ├── mcts_agent.py       # Saiyam: MCTS (Selection, Expansion, Simulation, Backprop)
│   └── q_learning_agent.py # Naman: Tabular Q-Learning, self-play, canonical D4 symmetries
├── tournament/
│   ├── match.py            # Head-to-head N-game series with alternating sides
│   └── benchmark.py        # Round-robin tournament engine & ASCII leaderboard
├── play.py                 # CLI match runner
├── run_tournament.py       # Tournament execution script
├── gui.py                  # Graphical UI visualizer
├── tests/
│   └── test_board.py       # Unit tests for board rules, bonus turns, captures, undo, symmetries
├── TEAM_GUIDE.md           # Technical guide & API reference for team members
└── README.md
```

---

## 🏆 Benchmark Leaderboard Sample

```
------------------------------------------------------------------------------------------------------------
                         TOURNAMENT LEADERBOARD
------------------------------------------------------------------------------------------------------------
Rank  | Agent Name                       | Games | W-L-D      | Win Rate  | Avg Pts | Avg Time   | Max Time 
------------------------------------------------------------------------------------------------------------
1     | Greedy Heuristic                 | 10    | 9-1-0      |  90.0%    |    7.10 |   0.01 ms  |    0.0 ms
2     | Rule-Based Expert (Ranjit)       | 10    | 8-2-0      |  80.0%    |    7.00 |   0.01 ms  |    0.0 ms
3     | Alpha-Beta Minimax (Shanmukh)    | 10    | 7-3-0      |  70.0%    |    6.30 |   0.38 ms  |    1.4 ms
4     | Monte Carlo Tree Search (Saiyam) | 10    | 3-7-0      |  30.0%    |    2.70 |  12.92 ms  |   98.5 ms
5     | Random Baseline                  | 10    | 2-8-0      |  20.0%    |    2.10 |   0.00 ms  |    0.0 ms
6     | Q-Learning RL (Naman)            | 10    | 1-9-0      |  10.0%    |    1.80 |   0.05 ms  |    0.1 ms
------------------------------------------------------------------------------------------------------------
```

---

## 🧪 Verification & Testing

Run unit tests via:
```bash
python3 -m unittest discover tests -v
```
Validates move legality, box completion, consecutive bonus turns, state rollback (`undo_move`), board cloning, and dihedral symmetry equivalences.
