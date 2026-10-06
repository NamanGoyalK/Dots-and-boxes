"""
Round-Robin Tournament & Benchmarking Engine for Dots and Boxes.
Executes games between all registered AI agents, computes metrics,
and renders formatted leaderboards and head-to-head cross tables.
"""

import json
import time
from typing import List, Dict, Any, Optional
from core.base_agent import BaseAgent
from tournament.match import play_series, SeriesResult


class Tournament:
    """
    Orchestrates round-robin matches among a list of BaseAgent instances.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        games_per_matchup: int = 10,
        rows: int = 3,
        cols: int = 3,
        time_limit_sec: float = 5.0,
        verbose: bool = False
    ):
        self.agents = agents
        self.games_per_matchup = games_per_matchup
        self.rows = rows
        self.cols = cols
        self.time_limit_sec = time_limit_sec
        self.verbose = verbose
        
        self.results: Dict[str, SeriesResult] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}

    def run(self) -> Dict[str, Any]:
        """
        Executes round-robin tournament for all agent pairs.
        """
        print(f"\n=======================================================")
        print(f"       DOTS AND BOXES BENCHMARK TOURNAMENT             ")
        print(f"=======================================================")
        print(f"Board Size: {self.rows}x{self.cols} ({self.rows * self.cols} boxes)")
        print(f"Agents ({len(self.agents)}): {[a.name for a in self.agents]}")
        print(f"Games per Matchup: {self.games_per_matchup} (alternating sides)")
        print(f"Move Time Limit: {self.time_limit_sec}s")
        print(f"=======================================================\n")

        # Initialize statistics for each agent
        for agent in self.agents:
            self.stats[agent.name] = {
                "agent": agent,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "total_points": 0,
                "total_opp_points": 0,
                "move_times": [],
                "h2h": {}  # opp_name -> {"wins": w, "losses": l, "draws": d}
            }

        total_matchups = (len(self.agents) * (len(self.agents) - 1)) // 2
        matchup_idx = 0
        start_time = time.perf_counter()

        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                matchup_idx += 1
                agent_a = self.agents[i]
                agent_b = self.agents[j]
                
                print(f"[{matchup_idx}/{total_matchups}] Running Matchup: {agent_a.name} vs {agent_b.name}...")
                
                series = play_series(
                    agent_a=agent_a,
                    agent_b=agent_b,
                    num_games=self.games_per_matchup,
                    rows=self.rows,
                    cols=self.cols,
                    time_limit_sec=self.time_limit_sec,
                    verbose=self.verbose
                )
                
                key = f"{agent_a.name}__vs__{agent_b.name}"
                self.results[key] = series

                # Update stats for agent A
                sa = self.stats[agent_a.name]
                sa["games"] += series.num_games
                sa["wins"] += series.a_wins
                sa["losses"] += series.b_wins
                sa["draws"] += series.draws
                sa["total_points"] += series.a_total_score
                sa["total_opp_points"] += series.b_total_score
                sa["move_times"].extend(series.a_move_times)
                sa["h2h"][agent_b.name] = {"wins": series.a_wins, "losses": series.b_wins, "draws": series.draws}

                # Update stats for agent B
                sb = self.stats[agent_b.name]
                sb["games"] += series.num_games
                sb["wins"] += series.b_wins
                sb["losses"] += series.a_wins
                sb["draws"] += series.draws
                sb["total_points"] += series.b_total_score
                sb["total_opp_points"] += series.a_total_score
                sb["move_times"].extend(series.b_move_times)
                sb["h2h"][agent_a.name] = {"wins": series.b_wins, "losses": series.a_wins, "draws": series.draws}

                print(f"    Result: {agent_a.name} [{series.a_wins}W - {series.b_wins}L - {series.draws}D] {agent_b.name}")

        total_elapsed = time.perf_counter() - start_time
        print(f"\nTournament completed in {total_elapsed:.2f} seconds.\n")

        # Compile and print results
        leaderboard = self._compile_leaderboard()
        self.print_leaderboard(leaderboard)
        self.print_h2h_matrix()

        return {
            "leaderboard": leaderboard,
            "duration_sec": total_elapsed,
            "rows": self.rows,
            "cols": self.cols
        }

    def _compile_leaderboard(self) -> List[Dict[str, Any]]:
        rows = []
        for name, s in self.stats.items():
            games = s["games"]
            wins = s["wins"]
            losses = s["losses"]
            draws = s["draws"]
            pts = s["total_points"]
            opp_pts = s["total_opp_points"]
            win_rate = (wins / games * 100.0) if games else 0.0
            avg_pts = (pts / games) if games else 0.0
            
            times = s["move_times"]
            avg_time_ms = (sum(times) / len(times) * 1000.0) if times else 0.0
            max_time_ms = (max(times) * 1000.0) if times else 0.0

            rows.append({
                "name": name,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
                "total_points": pts,
                "point_diff": pts - opp_pts,
                "avg_points": avg_pts,
                "avg_time_ms": avg_time_ms,
                "max_time_ms": max_time_ms
            })

        # Rank by win_rate descending, then point_diff descending
        rows.sort(key=lambda r: (r["win_rate"], r["point_diff"], r["total_points"]), reverse=True)
        for idx, row in enumerate(rows, 1):
            row["rank"] = idx
        return rows

    def print_leaderboard(self, leaderboard: List[Dict[str, Any]]):
        header = f"{'Rank':<5} | {'Agent Name':<32} | {'Games':<5} | {'W-L-D':<10} | {'Win Rate':<9} | {'Avg Pts':<7} | {'Avg Time':<10} | {'Max Time':<9}"
        sep = "-" * len(header)
        print(sep)
        print("                         TOURNAMENT LEADERBOARD")
        print(sep)
        print(header)
        print(sep)
        for r in leaderboard:
            wld = f"{r['wins']}-{r['losses']}-{r['draws']}"
            win_rate_str = f"{r['win_rate']:>5.1f}%"
            avg_time_str = f"{r['avg_time_ms']:>6.2f} ms"
            max_time_str = f"{r['max_time_ms']:>6.1f} ms"
            print(
                f"{r['rank']:<5} | {r['name']:<32} | {r['games']:<5} | {wld:<10} | "
                f"{win_rate_str:<9} | {r['avg_points']:>7.2f} | {avg_time_str:<10} | {max_time_str:<9}"
            )
        print(sep + "\n")

    def print_h2h_matrix(self):
        """Prints cross-table of pairwise head-to-head match results."""
        names = [a.name for a in self.agents]
        short_names = [f"A{i+1}" for i in range(len(names))]

        print("HEAD-TO-HEAD CROSS TABLE (Row vs Column: Wins-Losses-Draws)")
        print("-" * 65)
        # Legend
        for i, name in enumerate(names):
            print(f"  {short_names[i]}: {name}")
        print("-" * 65)

        col_header = "       " + " ".join([f"{sn:>10}" for sn in short_names])
        print(col_header)
        for i, n1 in enumerate(names):
            row_str = f"{short_names[i]:<5} |"
            for j, n2 in enumerate(names):
                if i == j:
                    row_str += f"{'--':>10} "
                else:
                    record = self.stats[n1]["h2h"].get(n2, {"wins": 0, "losses": 0, "draws": 0})
                    wld = f"{record['wins']}-{record['losses']}-{record['draws']}"
                    row_str += f"{wld:>10} "
            print(row_str)
        print("-" * 65 + "\n")

    def export_json(self, filepath: str):
        """Export tournament metrics to JSON."""
        leaderboard = self._compile_leaderboard()
        data = {
            "rows": self.rows,
            "cols": self.cols,
            "games_per_matchup": self.games_per_matchup,
            "leaderboard": leaderboard,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tournament results saved to {filepath}")
