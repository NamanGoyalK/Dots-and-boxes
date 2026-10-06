"""
Unit tests for core.board
"""

import unittest
from core.board import Board


class TestBoard(unittest.TestCase):
    def test_dimensions_and_edge_counts(self):
        # 3x3 board: 12 H edges + 12 V edges = 24 total edges, 9 boxes
        b = Board(3, 3)
        self.assertEqual(b.num_h_edges, 12)
        self.assertEqual(b.num_v_edges, 12)
        self.assertEqual(b.total_edges, 24)
        self.assertEqual(b.num_boxes, 9)
        self.assertEqual(len(b.get_valid_moves()), 24)

        # 2x2 board: 6 H edges + 6 V edges = 12 total edges, 4 boxes
        b2 = Board(2, 2)
        self.assertEqual(b2.num_h_edges, 6)
        self.assertEqual(b2.num_v_edges, 6)
        self.assertEqual(b2.total_edges, 12)
        self.assertEqual(b2.num_boxes, 4)

    def test_edge_coordinate_conversion(self):
        b = Board(3, 3)
        for e in range(b.total_edges):
            etype, r, c = b.id_to_edge(e)
            recovered_id = b.edge_to_id(etype, r, c)
            self.assertEqual(e, recovered_id)

    def test_single_box_completion_and_bonus_turn(self):
        b = Board(2, 2)
        # Box (0, 0) edges: H(0,0), H(1,0), V(0,0), V(0,1)
        top = b.edge_to_id('H', 0, 0)
        bottom = b.edge_to_id('H', 1, 0)
        left = b.edge_to_id('V', 0, 0)
        right = b.edge_to_id('V', 0, 1)

        # P1 plays top
        captured, extra_turn = b.make_move(top)
        self.assertEqual(captured, 0)
        self.assertFalse(extra_turn)
        self.assertEqual(b.current_player, 2)

        # P2 plays bottom
        captured, extra_turn = b.make_move(bottom)
        self.assertEqual(captured, 0)
        self.assertFalse(extra_turn)
        self.assertEqual(b.current_player, 1)

        # P1 plays left
        captured, extra_turn = b.make_move(left)
        self.assertEqual(captured, 0)
        self.assertFalse(extra_turn)
        self.assertEqual(b.current_player, 2)

        # Before right edge is played: box (0,0) has 3 edges
        self.assertEqual(b.get_box_edge_count(0, 0), 3)
        self.assertEqual(b.would_complete_box(right), 1)

        # P2 plays right edge and completes Box (0,0)!
        captured, extra_turn = b.make_move(right)
        self.assertEqual(captured, 1)
        self.assertTrue(extra_turn)
        self.assertEqual(b.current_player, 2)  # P2 gets bonus turn!
        self.assertEqual(b.scores[2], 1)
        self.assertEqual(b.scores[1], 0)
        self.assertEqual(b.box_owners[0][0], 2)

    def test_double_box_completion(self):
        """Test completing two boxes with a single edge."""
        b = Board(2, 2)
        # We complete 3 edges of Box(0,0) and 3 edges of Box(0,1)
        # Shared edge is V(0,1)
        # Box(0,0): H(0,0), H(1,0), V(0,0), [V(0,1)]
        # Box(0,1): H(0,1), H(1,1), V(0,2), [V(0,1)]
        b.make_move(b.edge_to_id('H', 0, 0))
        b.make_move(b.edge_to_id('H', 1, 0))
        b.make_move(b.edge_to_id('V', 0, 0))

        b.make_move(b.edge_to_id('H', 0, 1))
        b.make_move(b.edge_to_id('H', 1, 1))
        b.make_move(b.edge_to_id('V', 0, 2))

        shared_edge = b.edge_to_id('V', 0, 1)
        self.assertEqual(b.would_complete_box(shared_edge), 2)

        player_before = b.current_player
        captured, extra_turn = b.make_move(shared_edge)
        self.assertEqual(captured, 2)
        self.assertTrue(extra_turn)
        self.assertEqual(b.current_player, player_before)
        self.assertEqual(b.scores[player_before], 2)
        self.assertEqual(b.box_owners[0][0], player_before)
        self.assertEqual(b.box_owners[0][1], player_before)

    def test_undo_move(self):
        b = Board(2, 2)
        e1 = b.edge_to_id('H', 0, 0)
        e2 = b.edge_to_id('H', 1, 0)
        e3 = b.edge_to_id('V', 0, 0)
        e4 = b.edge_to_id('V', 0, 1)

        b.make_move(e1)
        b.make_move(e2)
        b.make_move(e3)
        b.make_move(e4)  # Completes box

        self.assertEqual(b.scores[2], 1)
        self.assertEqual(b.box_owners[0][0], 2)

        # Undo the capturing move
        b.undo_move()
        self.assertFalse(b.edges[e4])
        self.assertEqual(b.scores[2], 0)
        self.assertEqual(b.box_owners[0][0], 0)
        self.assertEqual(b.box_edge_counts[0][0], 3)
        self.assertEqual(b.current_player, 2)

        # Undo another move
        b.undo_move()
        self.assertFalse(b.edges[e3])
        self.assertEqual(b.box_edge_counts[0][0], 2)
        self.assertEqual(b.current_player, 1)

    def test_clone_independence(self):
        b = Board(2, 2)
        b.make_move(0)
        b.make_move(1)

        clone = b.clone()
        clone.make_move(2)

        self.assertTrue(clone.edges[2])
        self.assertFalse(b.edges[2])
        self.assertEqual(len(b.get_valid_moves()), 10)
        self.assertEqual(len(clone.get_valid_moves()), 9)

    def test_canonical_symmetries(self):
        b = Board(3, 3)
        key1 = b.get_canonical_state()
        b.make_move(b.edge_to_id('H', 0, 0))
        key2 = b.get_canonical_state()

        # Another board where the top-right corner is played
        b_rot = Board(3, 3)
        b_rot.make_move(b_rot.edge_to_id('H', 0, 2))
        key3 = b_rot.get_canonical_state()

        # Both single-corner moves are symmetrically equivalent!
        self.assertEqual(key2[0], key3[0])


if __name__ == '__main__':
    unittest.main()
