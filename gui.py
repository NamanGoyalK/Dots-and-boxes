"""
Graphical User Interface (GUI) for Dots and Boxes.
Built with standard Python Tkinter (no external GUI dependencies).
Supports interactive Human play and animated AI vs AI matches.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
from core.board import Board
from agents import (
    RandomAgent,
    GreedyAgent,
    RuleBasedAgent,
    MinimaxAgent,
    MCTSAgent,
    QLearningAgent
)

AI_FACTORY = {
    "Random Baseline": lambda: RandomAgent("Random Baseline"),
    "Greedy Heuristic": lambda: GreedyAgent("Greedy Heuristic"),
    "Rule-Based Expert (Ranjit)": lambda: RuleBasedAgent("Rule-Based Expert (Ranjit)"),
    "Alpha-Beta Minimax (Shanmukh)": lambda: MinimaxAgent("Alpha-Beta Minimax (Shanmukh)", max_depth=3),
    "Monte Carlo Tree Search (Saiyam)": lambda: MCTSAgent("Monte Carlo Tree Search (Saiyam)", iterations=150),
    "Q-Learning RL (Naman)": lambda: QLearningAgent("Q-Learning RL (Naman)", epsilon=0.0),
}


class DotsAndBoxesGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dots and Boxes - AI Arena")
        self.root.geometry("820x720")
        self.root.minsize(700, 600)
        self.root.configure(bg="#1e1e2e")

        # Game parameters
        self.rows = 3
        self.cols = 3
        self.board = Board(self.rows, self.cols)
        
        self.p1_type = "Human"
        self.p2_type = "Greedy"
        self.ai_agents = {1: None, 2: None}
        
        # Colors
        self.BG_COLOR = "#1e1e2e"
        self.PANEL_BG = "#282a36"
        self.DOT_COLOR = "#f8f8f2"
        self.LINE_EMPTY_COLOR = "#44475a"
        self.LINE_HOVER_COLOR = "#ffb86c"
        self.P1_COLOR = "#8be9fd"    # Cyan
        self.P2_COLOR = "#ff79c6"    # Pink
        self.P1_BOX_BG = "#1d3b53"
        self.P2_BOX_BG = "#4d2242"
        
        # Visual metrics
        self.canvas_width = 540
        self.canvas_height = 540
        self.padding = 60
        
        # Edge hitboxes mapping: tag/item_id -> edge_id
        self.line_items = {}
        self.hovered_edge = None
        self.is_running_ai_match = False

        self._build_ui()
        self._start_new_game()

    def _build_ui(self):
        # Top Header & Controls Panel
        ctrl_frame = tk.Frame(self.root, bg=self.PANEL_BG, padx=15, pady=10)
        ctrl_frame.pack(fill=tk.X, side=tk.TOP)

        # Title
        title = tk.Label(
            ctrl_frame,
            text="DOTS & BOXES AI PLATFORM",
            font=("Helvetica", 14, "bold"),
            fg="#50fa7b",
            bg=self.PANEL_BG
        )
        title.grid(row=0, column=0, columnspan=6, pady=(0, 8), sticky="w")

        # Player 1 selector
        tk.Label(ctrl_frame, text="Player 1:", fg=self.P1_COLOR, bg=self.PANEL_BG, font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", padx=4)
        self.p1_combo = ttk.Combobox(ctrl_frame, values=["Human"] + list(AI_FACTORY.keys()), width=30, state="readonly")
        self.p1_combo.set("Human")
        self.p1_combo.grid(row=1, column=1, padx=4)

        # Player 2 selector
        tk.Label(ctrl_frame, text="Player 2:", fg=self.P2_COLOR, bg=self.PANEL_BG, font=("Helvetica", 10, "bold")).grid(row=1, column=2, sticky="w", padx=4)
        self.p2_combo = ttk.Combobox(ctrl_frame, values=["Human"] + list(AI_FACTORY.keys()), width=30, state="readonly")
        self.p2_combo.set("Greedy Heuristic")
        self.p2_combo.grid(row=1, column=3, padx=4)

        # Board size
        tk.Label(ctrl_frame, text="Size:", fg="#f8f8f2", bg=self.PANEL_BG, font=("Helvetica", 10)).grid(row=1, column=4, sticky="w", padx=4)
        self.size_combo = ttk.Combobox(ctrl_frame, values=["2x2", "3x3", "4x4"], width=6, state="readonly")
        self.size_combo.set("3x3")
        self.size_combo.grid(row=1, column=5, padx=4)

        # Action Buttons
        btn_frame = tk.Frame(ctrl_frame, bg=self.PANEL_BG)
        btn_frame.grid(row=2, column=0, columnspan=6, pady=(8, 0), sticky="ew")

        start_btn = tk.Button(btn_frame, text="New Match", command=self._start_new_game, bg="#6272a4", fg="white", font=("Helvetica", 10, "bold"), padx=12)
        start_btn.pack(side=tk.LEFT, padx=4)

        # Score & Status Bar
        status_frame = tk.Frame(self.root, bg=self.BG_COLOR, pady=10)
        status_frame.pack(fill=tk.X)

        self.score_label = tk.Label(
            status_frame,
            text="P1: 0  |  P2: 0",
            font=("Helvetica", 13, "bold"),
            fg="#f8f8f2",
            bg=self.BG_COLOR
        )
        self.score_label.pack()

        self.turn_label = tk.Label(
            status_frame,
            text="Turn: Player 1",
            font=("Helvetica", 11),
            fg="#bd93f9",
            bg=self.BG_COLOR
        )
        self.turn_label.pack()

        # Canvas for the game board
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#21222c",
            highlightthickness=2,
            highlightbackground="#44475a"
        )
        self.canvas.pack(pady=10)

    def _start_new_game(self):
        self.is_running_ai_match = False
        size_str = self.size_combo.get()
        size = int(size_str[0])
        self.rows = size
        self.cols = size
        self.board = Board(self.rows, self.cols)

        # Set up agents
        self.p1_type = self.p1_combo.get()
        self.p2_type = self.p2_combo.get()
        
        self.ai_agents[1] = AI_FACTORY[self.p1_type]() if self.p1_type != "Human" else None
        self.ai_agents[2] = AI_FACTORY[self.p2_type]() if self.p2_type != "Human" else None

        self._draw_board()
        self._update_status()

        # Trigger AI move if P1 is AI
        self.root.after(200, self._check_ai_turn)

    def _compute_geometry(self):
        span_x = self.canvas_width - 2 * self.padding
        span_y = self.canvas_height - 2 * self.padding
        self.cell_w = span_x / self.cols
        self.cell_h = span_y / self.rows

    def _get_dot_pos(self, r: int, c: int):
        x = self.padding + c * self.cell_w
        y = self.padding + r * self.cell_h
        return x, y

    def _draw_board(self):
        self.canvas.delete("all")
        self.line_items.clear()
        self._compute_geometry()

        # 1. Draw Boxes (Background rectangles)
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = self._get_dot_pos(r, c)
                x2, y2 = self._get_dot_pos(r + 1, c + 1)
                owner = self.board.box_owners[r][c]
                bg_col = self.BG_COLOR
                text = ""
                if owner == 1:
                    bg_col = self.P1_BOX_BG
                    text = "P1"
                elif owner == 2:
                    bg_col = self.P2_BOX_BG
                    text = "P2"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg_col, outline="", tags="box")
                if text:
                    self.canvas.create_text((x1 + x2)/2, (y1 + y2)/2, text=text, fill=self.DOT_COLOR, font=("Helvetica", 14, "bold"))

        # 2. Draw Edges
        # Horizontal edges
        for r in range(self.rows + 1):
            for c in range(self.cols):
                edge_id = self.board.edge_to_id('H', r, c)
                x1, y1 = self._get_dot_pos(r, c)
                x2, y2 = self._get_dot_pos(r, c + 1)
                self._create_edge_item(edge_id, x1, y1, x2, y2)

        # Vertical edges
        for r in range(self.rows):
            for c in range(self.cols + 1):
                edge_id = self.board.edge_to_id('V', r, c)
                x1, y1 = self._get_dot_pos(r, c)
                x2, y2 = self._get_dot_pos(r + 1, c)
                self._create_edge_item(edge_id, x1, y1, x2, y2)

        # 3. Draw Dots on top
        dot_radius = 6
        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                x, y = self._get_dot_pos(r, c)
                self.canvas.create_oval(
                    x - dot_radius, y - dot_radius,
                    x + dot_radius, y + dot_radius,
                    fill=self.DOT_COLOR, outline=""
                )

    def _create_edge_item(self, edge_id: int, x1: float, y1: float, x2: float, y2: float):
        drawn = self.board.edges[edge_id]
        if drawn:
            color = self.P1_COLOR if self.board.current_player == 1 else self.P2_COLOR
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=5)
        else:
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill=self.LINE_EMPTY_COLOR, width=4)
            # Bind click and hover
            self.canvas.tag_bind(line_id, "<Button-1>", lambda event, e=edge_id: self._on_edge_click(e))
            self.canvas.tag_bind(line_id, "<Enter>", lambda event, l=line_id, e=edge_id: self._on_edge_hover(l, e))
            self.canvas.tag_bind(line_id, "<Leave>", lambda event, l=line_id, e=edge_id: self._on_edge_leave(l, e))
            
        self.line_items[edge_id] = line_id

    def _on_edge_hover(self, line_id, edge_id: int):
        if not self.board.edges[edge_id] and not self._is_current_player_ai():
            self.canvas.itemconfig(line_id, fill=self.LINE_HOVER_COLOR, width=5)

    def _on_edge_leave(self, line_id, edge_id: int):
        if not self.board.edges[edge_id]:
            self.canvas.itemconfig(line_id, fill=self.LINE_EMPTY_COLOR, width=4)

    def _on_edge_click(self, edge_id: int):
        if self._is_current_player_ai() or self.board.is_game_over():
            return
        self._apply_move(edge_id)

    def _apply_move(self, move: int):
        if not self.board.is_legal_move(move):
            return

        captured, extra_turn = self.board.make_move(move)
        self._draw_board()
        self._update_status()

        if self.board.is_game_over():
            self._handle_game_over()
        else:
            self.root.after(150, self._check_ai_turn)

    def _is_current_player_ai(self) -> bool:
        curr_p = self.board.current_player
        return self.ai_agents[curr_p] is not None

    def _check_ai_turn(self):
        if self.board.is_game_over():
            return
        curr_p = self.board.current_player
        ai = self.ai_agents[curr_p]
        if ai is not None:
            # Run AI move
            move = ai.get_move(self.board.clone())
            norm_move = self.board.normalize_move(move)
            self._apply_move(norm_move)

    def _update_status(self):
        p1_score = self.board.scores[1]
        p2_score = self.board.scores[2]
        self.score_label.config(
            text=f"P1 ({self.p1_type}): {p1_score}   |   P2 ({self.p2_type}): {p2_score}"
        )
        if not self.board.is_game_over():
            curr_p = self.board.current_player
            curr_name = self.p1_type if curr_p == 1 else self.p2_type
            color = self.P1_COLOR if curr_p == 1 else self.P2_COLOR
            self.turn_label.config(text=f"Turn: Player {curr_p} ({curr_name})", fg=color)

    def _handle_game_over(self):
        winner = self.board.get_winner()
        if winner == 1:
            msg = f"Player 1 ({self.p1_type}) Wins!"
        elif winner == 2:
            msg = f"Player 2 ({self.p2_type}) Wins!"
        else:
            msg = "It's a Draw!"
        self.turn_label.config(text=f"Game Over! {msg}", fg="#50fa7b")


def main():
    root = tk.Tk()
    app = DotsAndBoxesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
