"""
Unit tests for QLearningAgent in agents/q_learning_agent.py
"""

import os
import tempfile
import unittest
from core.board import Board
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from tournament.match import play_series


class TestQLearningAgent(unittest.TestCase):
    def test_canonical_action_transformation(self):
        """Test that canonical transformation correctly transforms actions."""
        board = Board(2, 2)
        agent = QLearningAgent(auto_load=False)

        state_key, trans = agent._get_state_and_trans(board)
        self.assertEqual(trans, (0, False))

        # Make a move and rotate
        board.make_move(0)
        state_key_1, trans_1 = agent._get_state_and_trans(board)
        can_action_1 = agent._map_action_to_canonical(board, 1, trans_1)
        self.assertTrue(0 <= can_action_1 < board.total_edges)

    def test_negamax_update_turn_inversion(self):
        """Test that when turn passes to opponent, future value is subtracted (negamax)."""
        agent = QLearningAgent(alpha=1.0, gamma=0.5, auto_load=False, use_symmetries=False, filter_candidates=False)
        board = Board(2, 2)
        
        # Next board has an action with Q = 4.0
        next_board = Board(2, 2)
        next_s = next_board.get_state_key()[0]
        agent.set_q_value(next_s, 0, 4.0)

        # Case 1: extra_turn is True (same player)
        agent.update_q(state_key=0, action=0, reward=2.0, next_board=next_board, extra_turn=True, done=False)
        # target = 2.0 + 0.5 * 4.0 = 4.0
        self.assertEqual(agent.get_q_value(0, 0), 4.0)

        # Case 2: extra_turn is False (turn passes to opponent, value inverted)
        agent.update_q(state_key=1, action=0, reward=0.0, next_board=next_board, extra_turn=False, done=False)
        # target = 0.0 - 0.5 * 4.0 = -2.0
        self.assertEqual(agent.get_q_value(1, 0), -2.0)

    def test_candidate_moves_filtering(self):
        """Test that candidate moves filter out blunders."""
        agent = QLearningAgent(auto_load=False, filter_candidates=True)
        board = Board(2, 2)
        
        # When all moves are safe, candidate moves should be safe moves
        candidates = agent.get_candidate_moves(board)
        self.assertEqual(len(candidates), len(board.get_safe_moves()))

        # Set up a board where Box (0,0) has 3 edges drawn (1 edge remaining to complete)
        board.make_move(board.edge_to_id('H', 0, 0))
        board.make_move(board.edge_to_id('H', 1, 0))
        board.make_move(board.edge_to_id('V', 0, 0))
        completing_edge = board.edge_to_id('V', 0, 1)

        candidates = agent.get_candidate_moves(board)
        # Should exclusively contain the completing edge
        self.assertIn(completing_edge, candidates)
        self.assertEqual(len(candidates), 1)

    def test_save_and_load_model(self):
        """Test model persistence to disk."""
        agent = QLearningAgent(auto_load=False)
        agent.set_q_value(12345, 2, 3.14)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            agent.save_model(tmp_path)
            
            new_agent = QLearningAgent(auto_load=False)
            new_agent.load_model(tmp_path)
            self.assertAlmostEqual(new_agent.get_q_value(12345, 2), 3.14)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_strong_performance_against_random(self):
        """Test that the loaded agent overwhelmingly defeats RandomAgent."""
        agent = QLearningAgent(auto_load=True, epsilon=0.0)
        random_agent = RandomAgent()

        series = play_series(agent, random_agent, num_games=20, rows=2, cols=2)
        # Agent should win at least 70% of games against random play
        win_rate = series.a_wins / series.num_games
        self.assertGreaterEqual(win_rate, 0.70, f"Expected win rate >= 70%, got {win_rate * 100:.1f}%")


if __name__ == '__main__':
    unittest.main()
