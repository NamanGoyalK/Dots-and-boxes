"""
Dots and Boxes - Core Board Engine
Supports arbitrary grid sizes (default: 3x3 boxes).
Features fast move execution, undo support for Minimax, cloning for MCTS,
and symmetry canonicalization for Reinforcement Learning.
"""

from typing import List, Tuple, Dict, Optional, Set
import copy


class Board:
    """
    Represents an R x C grid of boxes in Dots and Boxes.
    
    Grid Terminology:
    - rows: number of box rows (R)
    - cols: number of box columns (C)
    - Dots grid size: (rows + 1) x (cols + 1)
    - Horizontal edges: (rows + 1) * cols edges
    - Vertical edges: rows * (cols + 1) edges
    - Total edges: (rows + 1) * cols + rows * (cols + 1)
    
    Edges can be addressed either by an integer ID in [0, total_edges - 1]
    or by a tuple: ('H', r, c) or ('V', r, c).
    """

    def __init__(self, rows: int = 3, cols: int = 3):
        self.rows = rows
        self.cols = cols
        self.num_boxes = rows * cols
        
        self.num_h_edges = (rows + 1) * cols
        self.num_v_edges = rows * (cols + 1)
        self.total_edges = self.num_h_edges + self.num_v_edges
        
        # Precompute edge <-> box mappings for O(1) lookups
        self._precompute_mappings()
        
        # Game state
        self.edges = [False] * self.total_edges  # True if edge is drawn
        self.box_owners = [[0] * cols for _ in range(rows)]  # 0: empty, 1: P1, 2: P2
        self.box_edge_counts = [[0] * cols for _ in range(rows)]  # Number of drawn edges per box (0..4)
        
        self.current_player = 1  # 1 or 2
        self.scores = {1: 0, 2: 0}
        self.move_history: List[dict] = []
        
    def _precompute_mappings(self):
        """Precompute edge-to-box and box-to-edge relationships."""
        # box_to_edges[r][c] = list of 4 edge_ids: [top, bottom, left, right]
        self.box_to_edges: List[List[List[int]]] = [
            [[] for _ in range(self.cols)] for _ in range(self.rows)
        ]
        # edge_to_boxes[edge_id] = list of (r, c) box tuples bounded by this edge (at most 2)
        self.edge_to_boxes: List[List[Tuple[int, int]]] = [[] for _ in range(self.total_edges)]
        
        for r in range(self.rows):
            for c in range(self.cols):
                top_edge = self.edge_to_id('H', r, c)
                bottom_edge = self.edge_to_id('H', r + 1, c)
                left_edge = self.edge_to_id('V', r, c)
                right_edge = self.edge_to_id('V', r, c + 1)
                
                edges = [top_edge, bottom_edge, left_edge, right_edge]
                self.box_to_edges[r][c] = edges
                
                for e in edges:
                    if (r, c) not in self.edge_to_boxes[e]:
                        self.edge_to_boxes[e].append((r, c))

        # Precompute edge transformations for square boards
        self.sym_edge_map: Dict[Tuple[int, bool], List[int]] = {}
        if self.rows == self.cols:
            for rot in range(4):
                for flip in (False, True):
                    self.sym_edge_map[(rot, flip)] = [
                        self._compute_transform_edge(e, rot, flip)
                        for e in range(self.total_edges)
                    ]

    # --- Edge Coordinate Converters ---

    def edge_to_id(self, edge_type: str, r: int, c: int) -> int:
        """
        Convert ('H', r, c) or ('V', r, c) to flat edge ID.
        H: 0 <= r <= rows, 0 <= c < cols
        V: 0 <= r < rows,  0 <= c <= cols
        """
        edge_type = edge_type.upper()
        if edge_type == 'H':
            if not (0 <= r <= self.rows and 0 <= c < self.cols):
                raise ValueError(f"Invalid H-edge coordinates: ({r}, {c}) for {self.rows}x{self.cols} board")
            return r * self.cols + c
        elif edge_type == 'V':
            if not (0 <= r < self.rows and 0 <= c <= self.cols):
                raise ValueError(f"Invalid V-edge coordinates: ({r}, {c}) for {self.rows}x{self.cols} board")
            return self.num_h_edges + r * (self.cols + 1) + c
        else:
            raise ValueError(f"Unknown edge type: {edge_type}. Expected 'H' or 'V'.")

    def id_to_edge(self, edge_id: int) -> Tuple[str, int, int]:
        """Convert flat edge ID to ('H', r, c) or ('V', r, c)."""
        if not (0 <= edge_id < self.total_edges):
            raise ValueError(f"Edge ID {edge_id} out of bounds [0, {self.total_edges - 1}]")
        
        if edge_id < self.num_h_edges:
            r = edge_id // self.cols
            c = edge_id % self.cols
            return ('H', r, c)
        else:
            offset = edge_id - self.num_h_edges
            r = offset // (self.cols + 1)
            c = offset % (self.cols + 1)
            return ('V', r, c)

    def normalize_move(self, move) -> int:
        """Accepts either an integer edge ID or a tuple ('H'/'V', r, c) and returns int ID."""
        if isinstance(move, int):
            return move
        elif isinstance(move, (tuple, list)) and len(move) == 3:
            return self.edge_to_id(str(move[0]), int(move[1]), int(move[2]))
        raise TypeError(f"Move must be an int or a 3-tuple ('H'/'V', r, c), got: {type(move)} ({move})")

    # --- Core Game Rules & Moves ---

    def is_legal_move(self, move) -> bool:
        """Check if an edge is legal (within bounds and not yet drawn)."""
        try:
            edge_id = self.normalize_move(move)
            return 0 <= edge_id < self.total_edges and not self.edges[edge_id]
        except (ValueError, TypeError):
            return False

    def get_valid_moves(self) -> List[int]:
        """Return list of all remaining legal edge IDs."""
        return [i for i, drawn in enumerate(self.edges) if not drawn]

    def make_move(self, move) -> Tuple[int, bool]:
        """
        Execute a move for the current player.
        
        Args:
            move: int edge_id or ('H'/'V', r, c)
            
        Returns:
            (num_captured, extra_turn):
            - num_captured: number of boxes captured by this move (0, 1, or 2)
            - extra_turn: True if current player gets a bonus turn
            
        Raises:
            ValueError if the move is illegal or game is over.
        """
        if self.is_game_over():
            raise ValueError("Cannot make move: game is already over.")
            
        edge_id = self.normalize_move(move)
        if not (0 <= edge_id < self.total_edges):
            raise ValueError(f"Move {edge_id} out of range [0, {self.total_edges - 1}]")
        if self.edges[edge_id]:
            raise ValueError(f"Edge {edge_id} ({self.id_to_edge(edge_id)}) is already drawn.")
            
        # Draw edge
        self.edges[edge_id] = True
        current_p = self.current_player
        
        # Track captured boxes
        captured_boxes = []
        for (r, c) in self.edge_to_boxes[edge_id]:
            self.box_edge_counts[r][c] += 1
            if self.box_edge_counts[r][c] == 4:
                self.box_owners[r][c] = current_p
                captured_boxes.append((r, c))
                
        num_captured = len(captured_boxes)
        self.scores[current_p] += num_captured
        
        prev_player = current_p
        extra_turn = (num_captured > 0)
        
        # Update turn: bonus turn if completed a box, else turn passes to opponent
        if not extra_turn:
            self.current_player = 3 - current_p
            
        # Save record for undo_move
        record = {
            'edge_id': edge_id,
            'player': prev_player,
            'captured_boxes': captured_boxes,
            'next_player': self.current_player
        }
        self.move_history.append(record)
        
        return num_captured, extra_turn

    def undo_move(self) -> None:
        """
        Revert the last move played.
        Crucial for Minimax / Alpha-Beta search without deep copying.
        """
        if not self.move_history:
            raise ValueError("No moves to undo.")
            
        record = self.move_history.pop()
        edge_id = record['edge_id']
        player = record['player']
        captured_boxes = record['captured_boxes']
        
        # Undraw edge
        self.edges[edge_id] = False
        
        # Revert box captures
        for (r, c) in self.edge_to_boxes[edge_id]:
            self.box_edge_counts[r][c] -= 1
            
        for (r, c) in captured_boxes:
            self.box_owners[r][c] = 0
            self.scores[player] -= 1
            
        # Restore player turn
        self.current_player = player

    def is_game_over(self) -> bool:
        """Game is over when all boxes are captured or all edges drawn."""
        return (self.scores[1] + self.scores[2]) == self.num_boxes

    def get_winner(self) -> Optional[int]:
        """
        Returns:
            1 if Player 1 won,
            2 if Player 2 won,
            0 if tied/draw,
            None if game is still in progress.
        """
        if not self.is_game_over():
            return None
        if self.scores[1] > self.scores[2]:
            return 1
        elif self.scores[2] > self.scores[1]:
            return 2
        else:
            return 0  # Tie

    def clone(self) -> 'Board':
        """Create a deep copy of the board state for MCTS or tree search."""
        new_board = Board(self.rows, self.cols)
        new_board.edges = list(self.edges)
        new_board.box_owners = [list(row) for row in self.box_owners]
        new_board.box_edge_counts = [list(row) for row in self.box_edge_counts]
        new_board.current_player = self.current_player
        new_board.scores = dict(self.scores)
        # Note: move_history is omitted in clone for speed, as clone is used for forward rollouts
        return new_board

    # --- Heuristics and Helper Methods for Agents ---

    def get_box_edge_count(self, r: int, c: int) -> int:
        """Number of drawn sides for box (r, c) (0 to 4)."""
        return self.box_edge_counts[r][c]

    def get_box_missing_edges(self, r: int, c: int) -> List[int]:
        """Returns the unplayed edge IDs bounding box (r, c)."""
        return [e for e in self.box_to_edges[r][c] if not self.edges[e]]

    def would_complete_box(self, move) -> int:
        """
        Returns the number of boxes (0, 1, or 2) that would be completed
        if this edge was played right now.
        """
        edge_id = self.normalize_move(move)
        count = 0
        for (r, c) in self.edge_to_boxes[edge_id]:
            if self.box_edge_counts[r][c] == 3:
                count += 1
        return count

    def would_give_away_box(self, move) -> bool:
        """
        Returns True if playing this edge will make any adjacent box have 3 edges,
        meaning the opponent could complete it on their next turn.
        (Avoid unless forced!)
        """
        edge_id = self.normalize_move(move)
        for (r, c) in self.edge_to_boxes[edge_id]:
            if self.box_edge_counts[r][c] == 2:
                return True
        return False

    def get_safe_moves(self) -> List[int]:
        """
        Returns moves that:
        1. Complete a box (always good!), OR
        2. Do NOT give away a box (do not create a 3rd side).
        """
        safe = []
        for m in self.get_valid_moves():
            if self.would_complete_box(m) > 0 or not self.would_give_away_box(m):
                safe.append(m)
        return safe

    def get_capturable_moves(self) -> List[int]:
        """Returns all moves that immediately capture at least 1 box."""
        return [m for m in self.get_valid_moves() if self.would_complete_box(m) > 0]

    # --- State Hashing & Symmetries for Q-Learning ---

    def get_state_key(self) -> Tuple[int, int]:
        """
        Compact bitmask state representation for Q-table or hash map.
        Returns (edges_bitmask, current_player).
        """
        mask = 0
        for i, drawn in enumerate(self.edges):
            if drawn:
                mask |= (1 << i)
        return (mask, self.current_player)

    def get_canonical_transformation(self) -> Tuple[int, Tuple[int, bool]]:
        """
        For square boards (rows == cols), computes canonical representation
        by finding the minimum bitmask across all 8 dihedral symmetries (D4),
        and returns the transformation (rot, flip) that achieves it.
        Returns:
            (min_mask, (rot, flip))
        """
        if self.rows != self.cols:
            return self.get_state_key()[0], (0, False)
            
        min_mask = None
        best_trans = (0, False)
        
        for (rot, flip), trans_edges in self.sym_edge_map.items():
            mask = 0
            for e, drawn in enumerate(self.edges):
                if drawn:
                    mask |= (1 << trans_edges[e])
            if min_mask is None or mask < min_mask:
                min_mask = mask
                best_trans = (rot, flip)
                
        return min_mask, best_trans

    def get_canonical_state(self) -> Tuple[int, int]:
        """
        For square boards (rows == cols), computes canonical representation
        by finding the minimum bitmask across all 8 dihedral symmetries (D4).
        Collapses symmetrical game states by up to 8x for Q-Learning!
        """
        min_mask, _ = self.get_canonical_transformation()
        return (min_mask, self.current_player)

    def _get_symmetric_masks(self) -> List[int]:
        """Generate edge bitmasks for all 8 symmetries (4 rotations + reflections)."""
        if self.rows != self.cols:
            return [self.get_state_key()[0]]
        masks = []
        for (rot, flip), trans_edges in self.sym_edge_map.items():
            mask = 0
            for e, drawn in enumerate(self.edges):
                if drawn:
                    mask |= (1 << trans_edges[e])
            masks.append(mask)
        return masks

    def _compute_transform_edge(self, edge_id: int, rot: int, flip: bool) -> int:
        """Compute transform edge under rotation (0..3 * 90 deg) and horizontal flip."""
        etype, r, c = self.id_to_edge(edge_id)
        N = self.rows  # assuming rows == cols
        
        # Apply flip first (horizontal flip across vertical midline)
        if flip:
            if etype == 'H':
                c = (N - 1) - c
            else:
                c = N - c
                
        # Apply rotation (rot * 90 degrees clockwise)
        for _ in range(rot):
            if etype == 'H':
                # H(r, c) becomes V(c, N - r)
                etype = 'V'
                new_r = c
                new_c = N - r
                r, c = new_r, new_c
            else:
                # V(r, c) becomes H(c, N - 1 - r)
                etype = 'H'
                new_r = c
                new_c = N - 1 - r
                r, c = new_r, new_c
                
        return self.edge_to_id(etype, r, c)

    def _transform_edge(self, edge_id: int, rot: int, flip: bool) -> int:
        """Transform edge under rotation (0, 1, 2, 3 * 90 deg) and horizontal flip."""
        if hasattr(self, 'sym_edge_map') and (rot, flip) in self.sym_edge_map:
            return self.sym_edge_map[(rot, flip)][edge_id]
        return self._compute_transform_edge(edge_id, rot, flip)

    # --- Pretty Terminal Representation ---

    def render_str(self, show_indices: bool = False) -> str:
        """
        Renders the current board as an ASCII string.
        If show_indices is True, unplayed edges show their integer IDs for easy manual input.
        """
        lines = []
        for r in range(self.rows):
            # Horizontal row r
            h_row = []
            for c in range(self.cols):
                h_row.append("•")
                e_id = self.edge_to_id('H', r, c)
                if self.edges[e_id]:
                    h_row.append("───")
                else:
                    if show_indices:
                        h_row.append(f"{e_id:>3}")
                    else:
                        h_row.append("   ")
            h_row.append("•")
            lines.append("".join(h_row))
            
            # Vertical row r
            v_row = []
            for c in range(self.cols):
                e_id = self.edge_to_id('V', r, c)
                if self.edges[e_id]:
                    v_row.append("│")
                else:
                    if show_indices:
                        v_row.append(f"{e_id:>2}")
                    else:
                        v_row.append(" ")
                        
                owner = self.box_owners[r][c]
                box_str = f" {owner} " if owner > 0 else "   "
                v_row.append(box_str)
                
            # Last vertical edge of row r
            e_id = self.edge_to_id('V', r, self.cols)
            if self.edges[e_id]:
                v_row.append("│")
            else:
                if show_indices:
                    v_row.append(f"{e_id:>2}")
                else:
                    v_row.append(" ")
            lines.append("".join(v_row))
            
        # Bottom-most horizontal row (r = rows)
        last_h = []
        for c in range(self.cols):
            last_h.append("•")
            e_id = self.edge_to_id('H', self.rows, c)
            if self.edges[e_id]:
                last_h.append("───")
            else:
                if show_indices:
                    last_h.append(f"{e_id:>3}")
                else:
                    last_h.append("   ")
        last_h.append("•")
        lines.append("".join(last_h))
        
        status = f"Score: P1={self.scores[1]} | P2={self.scores[2]} | Turn: Player {self.current_player}"
        return "\n".join(lines) + "\n" + status
