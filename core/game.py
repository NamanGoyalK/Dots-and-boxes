"""
Game engine and match orchestrator for Dots and Boxes.
Executes games between two BaseAgent instances with timing, rule validation,
and detailed result reporting.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from core.board import Board
from core.base_agent import BaseAgent


@dataclass
class MatchResult:
    """Detailed summary of a completed match."""
    p1_name: str
    p2_name: str
    winner: int  # 1 for P1, 2 for P2, 0 for Draw, -1 for error/forfeit
    winner_name: str
    scores: Dict[int, int]
    total_moves: int
    duration_sec: float
    p1_move_times: List[float] = field(default_factory=list)
    p2_move_times: List[float] = field(default_factory=list)
    forfeit_by: Optional[int] = None
    forfeit_reason: Optional[str] = None
    history: List[dict] = field(default_factory=list)

    @property
    def p1_avg_time_ms(self) -> float:
        if not self.p1_move_times:
            return 0.0
        return (sum(self.p1_move_times) / len(self.p1_move_times)) * 1000.0

    @property
    def p2_avg_time_ms(self) -> float:
        if not self.p2_move_times:
            return 0.0
        return (sum(self.p2_move_times) / len(self.p2_move_times)) * 1000.0

    @property
    def p1_max_time_ms(self) -> float:
        if not self.p1_move_times:
            return 0.0
        return max(self.p1_move_times) * 1000.0

    @property
    def p2_max_time_ms(self) -> float:
        if not self.p2_move_times:
            return 0.0
        return max(self.p2_move_times) * 1000.0


class Game:
    """
    Runs a game of Dots and Boxes between two agents.
    """

    def __init__(
        self,
        agent1: BaseAgent,
        agent2: BaseAgent,
        rows: int = 3,
        cols: int = 3,
        time_limit_sec: Optional[float] = 5.0,
        verbose: bool = False
    ):
        self.agent1 = agent1
        self.agent2 = agent2
        self.agents = {1: agent1, 2: agent2}
        self.board = Board(rows=rows, cols=cols)
        self.time_limit_sec = time_limit_sec
        self.verbose = verbose

    def play(
        self,
        on_move_callback: Optional[Callable[[Board, int, Any], None]] = None
    ) -> MatchResult:
        """
        Executes the match to completion.
        
        Args:
            on_move_callback: Optional function(board, player, move) called after each move.
            
        Returns:
            MatchResult containing winner, scores, moves, and timing breakdown.
        """
        # Notify agents of match start
        self.agent1.on_game_start(1, self.board)
        self.agent2.on_game_start(2, self.board)
        
        p1_times: List[float] = []
        p2_times: List[float] = []
        history: List[dict] = []
        
        match_start_time = time.perf_counter()
        forfeit_by = None
        forfeit_reason = None

        if self.verbose:
            print(f"=== Match Start: {self.agent1.name} (P1) vs {self.agent2.name} (P2) ===")
            print(f"Board size: {self.board.rows}x{self.board.cols} ({self.board.total_edges} edges)\n")

        while not self.board.is_game_over():
            current_p = self.board.current_player
            agent = self.agents[current_p]
            
            # Pass a clone so agents cannot mutate the official game board
            board_view = self.board.clone()
            
            t0 = time.perf_counter()
            try:
                move = agent.get_move(board_view)
            except Exception as e:
                forfeit_by = current_p
                forfeit_reason = f"Agent {agent.name} raised exception: {str(e)}"
                if self.verbose:
                    print(f"ERROR: {forfeit_reason}")
                break
            t1 = time.perf_counter()
            elapsed = t1 - t0
            
            if current_p == 1:
                p1_times.append(elapsed)
            else:
                p2_times.append(elapsed)
                
            # Time limit check
            if self.time_limit_sec and elapsed > self.time_limit_sec:
                forfeit_by = current_p
                forfeit_reason = (
                    f"Agent {agent.name} exceeded time limit "
                    f"({elapsed:.3f}s > {self.time_limit_sec}s)"
                )
                if self.verbose:
                    print(f"TIME LIMIT EXCEEDED: {forfeit_reason}")
                break
                
            # Validation
            if not self.board.is_legal_move(move):
                forfeit_by = current_p
                forfeit_reason = f"Agent {agent.name} attempted illegal move: {move}"
                if self.verbose:
                    print(f"ILLEGAL MOVE: {forfeit_reason}")
                break
                
            norm_move = self.board.normalize_move(move)
            captured, extra_turn = self.board.make_move(norm_move)
            
            move_record = {
                'player': current_p,
                'move': norm_move,
                'edge_repr': self.board.id_to_edge(norm_move),
                'captured': captured,
                'time_sec': elapsed,
                'scores': dict(self.board.scores)
            }
            history.append(move_record)
            
            if self.verbose:
                print(
                    f"Player {current_p} ({agent.name}) -> Edge {norm_move} "
                    f"{self.board.id_to_edge(norm_move)} | Captured: {captured} | "
                    f"Score: P1={self.board.scores[1]} P2={self.board.scores[2]} "
                    f"{'(BONUS TURN!)' if extra_turn else ''}"
                )
                
            if on_move_callback:
                on_move_callback(self.board, current_p, norm_move)

        total_duration = time.perf_counter() - match_start_time

        # Determine winner
        if forfeit_by is not None:
            winner = 3 - forfeit_by  # Opponent wins
            winner_name = self.agents[winner].name + " (by forfeit)"
        else:
            winner = self.board.get_winner()
            if winner == 1:
                winner_name = self.agent1.name
            elif winner == 2:
                winner_name = self.agent2.name
            else:
                winner_name = "Draw"

        # End of game hooks
        self.agent1.on_game_end(winner, self.board.scores, self.board)
        self.agent2.on_game_end(winner, self.board.scores, self.board)

        result = MatchResult(
            p1_name=self.agent1.name,
            p2_name=self.agent2.name,
            winner=winner,
            winner_name=winner_name,
            scores=dict(self.board.scores),
            total_moves=len(history),
            duration_sec=total_duration,
            p1_move_times=p1_times,
            p2_move_times=p2_times,
            forfeit_by=forfeit_by,
            forfeit_reason=forfeit_reason,
            history=history
        )

        if self.verbose:
            print("\n=== Match Ended ===")
            print(f"Winner: {winner_name}")
            print(f"Final Score: P1 {self.board.scores[1]} - {self.board.scores[2]} P2")
            print(f"Moves: {len(history)} | Duration: {total_duration:.3f}s\n")

        return result
