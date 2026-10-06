"""
Head-to-head match series between two agents.
Alternates who plays as Player 1 (first mover) to eliminate first-player bias.
"""

from typing import Dict, List
from core.base_agent import BaseAgent
from core.game import Game, MatchResult


class SeriesResult:
    """Aggregated statistics for a multi-game series between two agents."""

    def __init__(self, agent_a: BaseAgent, agent_b: BaseAgent, num_games: int):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.num_games = num_games
        
        self.a_wins = 0
        self.b_wins = 0
        self.draws = 0
        self.a_total_score = 0
        self.b_total_score = 0
        
        self.a_move_times: List[float] = []
        self.b_move_times: List[float] = []
        self.match_results: List[MatchResult] = []

    def record_match(self, match: MatchResult, a_is_p1: bool):
        self.match_results.append(match)
        
        if a_is_p1:
            a_score = match.scores[1]
            b_score = match.scores[2]
            self.a_move_times.extend(match.p1_move_times)
            self.b_move_times.extend(match.p2_move_times)
            if match.winner == 1:
                self.a_wins += 1
            elif match.winner == 2:
                self.b_wins += 1
            else:
                self.draws += 1
        else:
            a_score = match.scores[2]
            b_score = match.scores[1]
            self.a_move_times.extend(match.p2_move_times)
            self.b_move_times.extend(match.p1_move_times)
            if match.winner == 2:
                self.a_wins += 1
            elif match.winner == 1:
                self.b_wins += 1
            else:
                self.draws += 1

        self.a_total_score += a_score
        self.b_total_score += b_score

    @property
    def a_win_rate(self) -> float:
        return (self.a_wins / self.num_games) * 100.0 if self.num_games else 0.0

    @property
    def b_win_rate(self) -> float:
        return (self.b_wins / self.num_games) * 100.0 if self.num_games else 0.0

    @property
    def a_avg_score(self) -> float:
        return self.a_total_score / self.num_games if self.num_games else 0.0

    @property
    def b_avg_score(self) -> float:
        return self.b_total_score / self.num_games if self.num_games else 0.0

    @property
    def a_avg_time_ms(self) -> float:
        if not self.a_move_times:
            return 0.0
        return (sum(self.a_move_times) / len(self.a_move_times)) * 1000.0

    @property
    def b_avg_time_ms(self) -> float:
        if not self.b_move_times:
            return 0.0
        return (sum(self.b_move_times) / len(self.b_move_times)) * 1000.0


def play_series(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    num_games: int = 10,
    rows: int = 3,
    cols: int = 3,
    time_limit_sec: float = 5.0,
    verbose: bool = False
) -> SeriesResult:
    """
    Plays an N-game series alternating Player 1 / Player 2 roles.
    """
    series = SeriesResult(agent_a, agent_b, num_games)

    for game_idx in range(num_games):
        # Alternate sides: even games -> A is P1; odd games -> B is P1
        a_is_p1 = (game_idx % 2 == 0)
        p1 = agent_a if a_is_p1 else agent_b
        p2 = agent_b if a_is_p1 else agent_a

        game = Game(
            agent1=p1,
            agent2=p2,
            rows=rows,
            cols=cols,
            time_limit_sec=time_limit_sec,
            verbose=verbose
        )
        result = game.play()
        series.record_match(result, a_is_p1=a_is_p1)

    return series
