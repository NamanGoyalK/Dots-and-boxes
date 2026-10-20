"""
Dots & Boxes
Minimalist Monochrome Desktop Application with Deep Pink Accent.
Zero external GUI dependencies (Python standard Tkinter).
"""

import math
import tkinter as tk
from tkinter import ttk, font as tkfont
from typing import Optional, Tuple, Dict

from core.board import Board
from agents import (
    RandomAgent,
    GreedyAgent,
    RuleBasedAgent,
    MinimaxAgent,
    MCTSAgent,
    QLearningAgent,
)

try:
    from agents.dqn_agent import DQNAgent, TORCH_AVAILABLE
except Exception:
    TORCH_AVAILABLE = False


SHORT_NAMES = {
    "Human": "Human",
    "Random Baseline": "Random",
    "Greedy Heuristic": "Greedy",
    "Rule-Based Expert (Ranjit)": "Rule-Based",
    "Alpha-Beta Minimax (Shanmukh)": "Minimax",
    "Monte Carlo Tree Search (Saiyam)": "MCTS",
    "Q-Learning RL (Naman)": "Q-Learning",
    "Deep Q-Network (GPU/CUDA)": "DQN",
}

AI_FACTORY = {
    "Random Baseline": lambda r, c: RandomAgent("Random Baseline"),
    "Greedy Heuristic": lambda r, c: GreedyAgent("Greedy Heuristic"),
    "Rule-Based Expert (Ranjit)": lambda r, c: RuleBasedAgent(
        "Rule-Based Expert (Ranjit)"
    ),
    "Alpha-Beta Minimax (Shanmukh)": lambda r, c: MinimaxAgent(
        "Alpha-Beta Minimax (Shanmukh)", max_depth=4 if r <= 2 else (3 if r == 3 else 2)
    ),
    "Monte Carlo Tree Search (Saiyam)": lambda r, c: MCTSAgent(
        "Monte Carlo Tree Search (Saiyam)", iterations=200 if r <= 3 else 100
    ),
    "Q-Learning RL (Naman)": lambda r, c: QLearningAgent(
        "Q-Learning RL (Naman)", epsilon=0.0
    ),
}

if TORCH_AVAILABLE:
    AI_FACTORY["Deep Q-Network (GPU/CUDA)"] = lambda r, c: DQNAgent(
        name="Deep Q-Network (GPU/CUDA)", rows=r, cols=c
    )


class DotsAndBoxesGUI:
    # Minimalist Monochrome + Deep Pink Accent Palette
    BG_WINDOW = "#09090b"  # Deep Obsidian
    BG_SURFACE = "#121214"  # Top & sidebar panels
    BG_CARD = "#18181b"  # Metric cards
    BORDER_HAIRLINE = "#27272a"  # 1px border
    CANVAS_BG = "#0c0c0e"  # Matte black canvas

    TEXT_PRIMARY = "#f4f4f5"  # Stark Chalk White
    TEXT_SECONDARY = "#a1a1aa"  # Muted Gray
    TEXT_MUTED = "#71717a"  # Dim Gray

    # Player 1: Chalk White
    P1_COLOR = "#babad7"
    P1_BOX_BG = "#292930"
    P1_BOX_BORDER = "#3e3e4a"

    # Player 2: Warm Dark Golden Yellow Accent
    P2_COLOR = "#eaa208"  # Warm Dark Golden Yellow (yellowish, not orange)
    P2_BOX_BG = "#201c10"  # Warm Gold-Infused Charcoal
    P2_BOX_BORDER = "#ea9708"

    # Board Elements
    DOT_COLOR = "#f4f4f5"
    DOT_OUTLINE = "#27272a"
    LINE_EMPTY = "#1c1c1f"  # Faint hairline track
    LINE_HOVER = "#71717a"  # Ghost white hover
    LINE_HINT = "#10b981"  # Subtle emerald hint
    LINE_LAST_MOVE = "#fee78a"  # Soft luminous gold highlight for last move

    SNAP_RADIUS = 26

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dots & Boxes")
        self.root.geometry("1040x760")
        self.root.minsize(900, 660)
        self.root.configure(bg=self.BG_WINDOW)

        self.rows = 3
        self.cols = 3
        self.board = Board(self.rows, self.cols)

        self.p1_type = "Human"
        self.p2_type = "Greedy Heuristic"
        self.ai_agents: Dict[int, Optional[object]] = {1: None, 2: None}
        self.edge_drawn_by: Dict[int, int] = {}

        self.session_scores = {1: 0, 2: 0, "draw": 0}
        self.last_move: Optional[int] = None
        self.suggested_move: Optional[int] = None
        self.hovered_edge: Optional[int] = None

        self.is_ai_running = False
        self.ai_delay_ms = 250
        self.ai_job = None
        self.game_over_card_visible = False

        self.canvas_width = 540
        self.canvas_height = 540
        self.padding = 60
        self.cell_w = 0.0
        self.cell_h = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.edge_coords: Dict[int, Tuple[float, float, float, float]] = {}

        self._configure_styles()
        self._build_ui()
        self._bind_shortcuts()
        self._start_new_game()

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "TCombobox",
            fieldbackground=self.BG_CARD,
            background=self.BG_SURFACE,
            foreground=self.TEXT_PRIMARY,
            darkcolor=self.BORDER_HAIRLINE,
            lightcolor=self.BORDER_HAIRLINE,
            arrowcolor=self.TEXT_SECONDARY,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.BG_CARD)],
            selectbackground=[("readonly", self.BG_CARD)],
            selectforeground=[("readonly", self.TEXT_PRIMARY)],
        )

    def _build_ui(self):
        # Header Bar
        header = tk.Frame(
            self.root,
            bg=self.BG_SURFACE,
            padx=20,
            pady=12,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        header.pack(fill=tk.X, side=tk.TOP)

        # Title
        lbl_title = tk.Label(
            header,
            text="Dots & Boxes",
            font=("Helvetica", 12, "bold"),
            fg=self.TEXT_PRIMARY,
            bg=self.BG_SURFACE,
        )
        lbl_title.pack(side=tk.LEFT, padx=(0, 28))

        # Match Setup
        ctrl_box = tk.Frame(header, bg=self.BG_SURFACE)
        ctrl_box.pack(side=tk.LEFT, expand=True)

        tk.Label(
            ctrl_box,
            text="P1",
            fg=self.P1_COLOR,
            bg=self.BG_SURFACE,
            font=("Helvetica", 9, "bold"),
        ).grid(row=0, column=0, padx=(0, 6), sticky="e")
        self.p1_combo = ttk.Combobox(
            ctrl_box,
            values=["Human"] + list(AI_FACTORY.keys()),
            width=20,
            state="readonly",
        )
        self.p1_combo.set("Human")
        self.p1_combo.grid(row=0, column=1, padx=(0, 16))

        tk.Label(
            ctrl_box,
            text="P2",
            fg=self.P2_COLOR,
            bg=self.BG_SURFACE,
            font=("Helvetica", 9, "bold"),
        ).grid(row=0, column=2, padx=(8, 6), sticky="e")
        self.p2_combo = ttk.Combobox(
            ctrl_box,
            values=["Human"] + list(AI_FACTORY.keys()),
            width=20,
            state="readonly",
        )
        self.p2_combo.set("Greedy Heuristic")
        self.p2_combo.grid(row=0, column=3, padx=(0, 16))

        tk.Label(
            ctrl_box,
            text="Grid",
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
            font=("Helvetica", 9),
        ).grid(row=0, column=4, padx=(8, 6), sticky="e")
        self.size_combo = ttk.Combobox(
            ctrl_box, values=["2x2", "3x3", "4x4", "5x5"], width=6, state="readonly"
        )
        self.size_combo.set("3x3")
        self.size_combo.grid(row=0, column=5, padx=(0, 12))
        self.size_combo.bind("<<ComboboxSelected>>", lambda e: self._start_new_game())

        # New Game Button
        btn_new = tk.Button(
            header,
            text="New Game",
            command=self._start_new_game,
            bg=self.TEXT_PRIMARY,
            fg=self.BG_WINDOW,
            activebackground="#ffffff",
            activeforeground=self.BG_WINDOW,
            font=("Helvetica", 9, "bold"),
            padx=14,
            pady=4,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
        )
        btn_new.pack(side=tk.RIGHT)

        # Main Layout
        main_container = tk.Frame(self.root, bg=self.BG_WINDOW, padx=20, pady=16)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left Column: Board
        left_col = tk.Frame(main_container, bg=self.BG_WINDOW)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Status Strip
        self.status_banner = tk.Frame(
            left_col,
            bg=self.BG_CARD,
            padx=14,
            pady=8,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        self.status_banner.pack(fill=tk.X, pady=(0, 10))

        self.turn_pill = tk.Label(
            self.status_banner,
            text="Player 1's Turn",
            font=("Helvetica", 10, "bold"),
            fg=self.TEXT_PRIMARY,
            bg=self.BG_CARD,
        )
        self.turn_pill.pack(side=tk.LEFT)

        self.hint_status_lbl = tk.Label(
            self.status_banner,
            text="Select an edge",
            font=("Helvetica", 9),
            fg=self.TEXT_MUTED,
            bg=self.BG_CARD,
        )
        self.hint_status_lbl.pack(side=tk.RIGHT)

        # Board Canvas
        self.canvas_frame = tk.Frame(
            left_col,
            bg=self.CANVAS_BG,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.CANVAS_BG,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

        # Bottom Controls
        bot_strip = tk.Frame(
            left_col,
            bg=self.BG_SURFACE,
            padx=12,
            pady=8,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        bot_strip.pack(fill=tk.X, pady=(10, 0))

        def make_btn(parent, text, cmd):
            return tk.Button(
                parent,
                text=text,
                command=cmd,
                bg=self.BG_CARD,
                fg=self.TEXT_PRIMARY,
                activebackground=self.BORDER_HAIRLINE,
                activeforeground=self.TEXT_PRIMARY,
                font=("Helvetica", 8, "bold"),
                padx=10,
                pady=3,
                relief=tk.FLAT,
                bd=0,
                highlightbackground=self.BORDER_HAIRLINE,
                highlightthickness=1,
                cursor="hand2",
            )

        self.btn_play_pause = make_btn(bot_strip, "Run AI", self._toggle_ai_playback)
        self.btn_play_pause.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_step = make_btn(bot_strip, "Step", self._step_ai_move)
        self.btn_step.pack(side=tk.LEFT, padx=6)

        self.btn_undo = make_btn(bot_strip, "Undo", self._undo_move)
        self.btn_undo.pack(side=tk.LEFT, padx=6)

        self.btn_hint = make_btn(bot_strip, "Hint", self._show_hint)
        self.btn_hint.pack(side=tk.LEFT, padx=6)

        tk.Label(
            bot_strip,
            text="Speed",
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
            font=("Helvetica", 8),
        ).pack(side=tk.LEFT, padx=(18, 6))
        self.speed_combo = ttk.Combobox(
            bot_strip,
            values=["Fast (80ms)", "Normal (250ms)", "Slow (600ms)"],
            width=14,
            state="readonly",
        )
        self.speed_combo.set("Normal (250ms)")
        self.speed_combo.pack(side=tk.LEFT)
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Right Column: Clean Sidebar
        right_col = tk.Frame(
            main_container,
            bg=self.BG_SURFACE,
            width=290,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))
        right_col.pack_propagate(False)

        # Score Section
        score_frame = tk.Frame(right_col, bg=self.BG_SURFACE, padx=14, pady=12)
        score_frame.pack(fill=tk.X)

        score_row = tk.Frame(score_frame, bg=self.BG_SURFACE)
        score_row.pack(fill=tk.X, pady=(0, 8))

        p1_pod = tk.Frame(
            score_row,
            bg=self.BG_CARD,
            padx=10,
            pady=8,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        p1_pod.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.p1_name_lbl = tk.Label(
            p1_pod,
            text="P1 · Human",
            fg=self.P1_COLOR,
            bg=self.BG_CARD,
            font=("Helvetica", 8, "bold"),
        )
        self.p1_name_lbl.pack()
        self.p1_score_val = tk.Label(
            p1_pod,
            text="0",
            fg=self.P1_COLOR,
            bg=self.BG_CARD,
            font=("Helvetica", 22, "bold"),
        )
        self.p1_score_val.pack()

        p2_pod = tk.Frame(
            score_row,
            bg=self.BG_CARD,
            padx=10,
            pady=8,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        p2_pod.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(4, 0))
        self.p2_name_lbl = tk.Label(
            p2_pod,
            text="P2 · Greedy",
            fg=self.P2_COLOR,
            bg=self.BG_CARD,
            font=("Helvetica", 8, "bold"),
        )
        self.p2_name_lbl.pack()
        self.p2_score_val = tk.Label(
            p2_pod,
            text="0",
            fg=self.P2_COLOR,
            bg=self.BG_CARD,
            font=("Helvetica", 22, "bold"),
        )
        self.p2_score_val.pack()

        # Progress bar
        self.progress_canvas = tk.Canvas(
            score_frame, height=4, bg=self.BG_CARD, highlightthickness=0
        )
        self.progress_canvas.pack(fill=tk.X, pady=(4, 6))

        self.session_lbl = tk.Label(
            score_frame,
            text="P1: 0  ·  P2: 0  ·  Draws: 0",
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
            font=("Helvetica", 8),
        )
        self.session_lbl.pack()

        # Metrics Section
        metrics_frame = tk.Frame(
            right_col,
            bg=self.BG_SURFACE,
            padx=14,
            pady=8,
            highlightbackground=self.BORDER_HAIRLINE,
            highlightthickness=1,
        )
        metrics_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.tactical_boxes_lbl = tk.Label(
            metrics_frame,
            text="Boxes: 0 / 9",
            fg=self.TEXT_SECONDARY,
            bg=self.BG_SURFACE,
            font=("Helvetica", 8),
            anchor="w",
        )
        self.tactical_boxes_lbl.pack(fill=tk.X, pady=1)

        self.tactical_edges_lbl = tk.Label(
            metrics_frame,
            text="Edges: 24 / 24",
            fg=self.TEXT_SECONDARY,
            bg=self.BG_SURFACE,
            font=("Helvetica", 8),
            anchor="w",
        )
        self.tactical_edges_lbl.pack(fill=tk.X, pady=1)

        self.tactical_critical_lbl = tk.Label(
            metrics_frame,
            text="Critical (3-sided): 0",
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
            font=("Helvetica", 8),
            anchor="w",
        )
        self.tactical_critical_lbl.pack(fill=tk.X, pady=1)

        # Move Log Section
        log_frame = tk.Frame(right_col, bg=self.BG_SURFACE, padx=10, pady=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            log_frame,
            text="History",
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
            font=("Helvetica", 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.move_log = tk.Listbox(
            log_frame,
            bg=self.BG_CARD,
            fg=self.TEXT_SECONDARY,
            selectbackground=self.BORDER_HAIRLINE,
            selectforeground=self.TEXT_PRIMARY,
            font=("Monaco", 8) if "Monaco" in tkfont.families() else ("Consolas", 8),
            highlightthickness=0,
            bd=0,
            yscrollcommand=log_scroll.set,
        )
        self.move_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.move_log.yview)

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self._on_space_pressed())
        self.root.bind("<u>", lambda e: self._undo_move())
        self.root.bind("<Control-z>", lambda e: self._undo_move())
        self.root.bind("<h>", lambda e: self._show_hint())
        self.root.bind("<r>", lambda e: self._start_new_game())

    def _on_space_pressed(self):
        if self.board.is_game_over():
            self._start_new_game()
        else:
            self._toggle_ai_playback()

    def _on_speed_change(self, event=None):
        val = self.speed_combo.get()
        if "Fast" in val:
            self.ai_delay_ms = 80
        elif "Slow" in val:
            self.ai_delay_ms = 600
        else:
            self.ai_delay_ms = 250

    def _on_canvas_resize(self, event):
        w = max(event.width, 300)
        h = max(event.height, 300)
        if abs(w - self.canvas_width) > 4 or abs(h - self.canvas_height) > 4:
            self.canvas_width = w
            self.canvas_height = h
            self._draw_board()

    # Game Lifecycle
    def _start_new_game(self):
        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None

        self.is_ai_running = False
        self.btn_play_pause.config(text="Run AI")
        self.game_over_card_visible = False

        size_str = self.size_combo.get()
        size = int(size_str[0])
        self.rows = size
        self.cols = size
        self.board = Board(self.rows, self.cols)

        self.edge_drawn_by.clear()
        self.last_move = None
        self.suggested_move = None
        self.hovered_edge = None

        self.p1_type = self.p1_combo.get()
        self.p2_type = self.p2_combo.get()

        self.ai_agents[1] = (
            AI_FACTORY[self.p1_type](self.rows, self.cols)
            if self.p1_type != "Human"
            else None
        )
        self.ai_agents[2] = (
            AI_FACTORY[self.p2_type](self.rows, self.cols)
            if self.p2_type != "Human"
            else None
        )

        for pid in (1, 2):
            if self.ai_agents[pid] is not None:
                self.ai_agents[pid].on_game_start(pid, self.board)

        self.move_log.delete(0, tk.END)
        self.hint_status_lbl.config(text="Select an edge", fg=self.TEXT_MUTED)
        self._update_analytics()
        self._draw_board()

        if self._is_current_player_ai():
            self._schedule_next_ai_step(delay=300)

    def _compute_geometry(self):
        avail_w = max(self.canvas_width - 2 * self.padding, 100)
        avail_h = max(self.canvas_height - 2 * self.padding, 100)
        grid_size = min(avail_w, avail_h)

        self.cell_w = grid_size / self.cols
        self.cell_h = grid_size / self.rows

        self.offset_x = (self.canvas_width - grid_size) / 2
        self.offset_y = (self.canvas_height - grid_size) / 2

        self.edge_coords.clear()

        for r in range(self.rows + 1):
            for c in range(self.cols):
                e = self.board.edge_to_id("H", r, c)
                x1, y1 = self._get_dot_pos(r, c)
                x2, y2 = self._get_dot_pos(r, c + 1)
                self.edge_coords[e] = (x1, y1, x2, y2)

        for r in range(self.rows):
            for c in range(self.cols + 1):
                e = self.board.edge_to_id("V", r, c)
                x1, y1 = self._get_dot_pos(r, c)
                x2, y2 = self._get_dot_pos(r + 1, c)
                self.edge_coords[e] = (x1, y1, x2, y2)

    def _get_dot_pos(self, r: int, c: int) -> Tuple[float, float]:
        x = self.offset_x + c * self.cell_w
        y = self.offset_y + r * self.cell_h
        return x, y

    def _dist_to_segment(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        if x1 == x2:
            min_y, max_y = min(y1, y2), max(y1, y2)
            if min_y <= py <= max_y:
                return abs(px - x1)
            elif py < min_y:
                return math.hypot(px - x1, py - min_y)
            else:
                return math.hypot(px - x1, py - max_y)
        else:
            min_x, max_x = min(x1, x2), max(x1, x2)
            if min_x <= px <= max_x:
                return abs(py - y1)
            elif px < min_x:
                return math.hypot(px - min_x, py - y1)
            else:
                return math.hypot(px - max_x, py - y1)

    def _get_closest_undrawn_edge(
        self, mx: float, my: float
    ) -> Tuple[Optional[int], float]:
        best_edge = None
        best_dist = float("inf")
        for e, (x1, y1, x2, y2) in self.edge_coords.items():
            if not self.board.edges[e]:
                d = self._dist_to_segment(mx, my, x1, y1, x2, y2)
                if d < best_dist:
                    best_dist = d
                    best_edge = e
        return best_edge, best_dist

    # Mouse Events
    def _on_mouse_move(self, event):
        if self._is_current_player_ai() or self.board.is_game_over():
            if self.hovered_edge is not None:
                self.hovered_edge = None
                self.canvas.config(cursor="")
                self._draw_board()
            return

        closest, dist = self._get_closest_undrawn_edge(event.x, event.y)
        if dist <= self.SNAP_RADIUS:
            if self.hovered_edge != closest:
                self.hovered_edge = closest
                self.canvas.config(cursor="hand2")
                self._draw_board()
        else:
            if self.hovered_edge is not None:
                self.hovered_edge = None
                self.canvas.config(cursor="")
                self._draw_board()

    def _on_mouse_leave(self, event):
        if self.hovered_edge is not None:
            self.hovered_edge = None
            self.canvas.config(cursor="")
            self._draw_board()

    def _on_canvas_click(self, event):
        if self.board.is_game_over() and self.game_over_card_visible:
            self._start_new_game()
            return

        if self._is_current_player_ai() or self.board.is_game_over():
            return

        target_edge = self.hovered_edge
        if target_edge is None:
            closest, dist = self._get_closest_undrawn_edge(event.x, event.y)
            if dist <= self.SNAP_RADIUS + 8:
                target_edge = closest

        if target_edge is not None and not self.board.edges[target_edge]:
            self.suggested_move = None
            self._apply_move(target_edge)

    # Moves & Game Progression
    def _apply_move(self, move: int):
        if not self.board.is_legal_move(move):
            return

        curr_player = self.board.current_player
        self.edge_drawn_by[move] = curr_player
        self.last_move = move

        edge_tag = self.board.id_to_edge(move)
        captured, extra_turn = self.board.make_move(move)

        tag_str = f"{edge_tag[0]}({edge_tag[1]},{edge_tag[2]})"
        if captured > 0:
            log_text = f"P{curr_player}: {tag_str} (+{captured})"
        else:
            log_text = f"P{curr_player}: {tag_str}"

        self.move_log.insert(tk.END, log_text)
        self.move_log.see(tk.END)

        self._update_analytics()
        self._draw_board()

        if self.board.is_game_over():
            self._handle_game_over()
        else:
            if self._is_current_player_ai():
                self._schedule_next_ai_step(delay=self.ai_delay_ms)

    def _is_current_player_ai(self) -> bool:
        curr_p = self.board.current_player
        return self.ai_agents[curr_p] is not None

    def _schedule_next_ai_step(self, delay: int = 150):
        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
        self.ai_job = self.root.after(delay, self._step_ai_move)

    def _step_ai_move(self):
        self.ai_job = None
        if self.board.is_game_over():
            return

        curr_p = self.board.current_player
        ai = self.ai_agents[curr_p]
        if ai is None:
            return

        try:
            move = ai.get_move(self.board.clone())
            norm_move = self.board.normalize_move(move)
            self._apply_move(norm_move)
        except Exception as e:
            self.hint_status_lbl.config(text=f"AI Error: {e}", fg=self.P2_COLOR)

    def _toggle_ai_playback(self):
        if self.board.is_game_over():
            self._start_new_game()
            return

        if self.is_ai_running:
            self.is_ai_running = False
            self.btn_play_pause.config(text="Run AI")
            if self.ai_job is not None:
                self.root.after_cancel(self.ai_job)
                self.ai_job = None
        else:
            self.is_ai_running = True
            self.btn_play_pause.config(text="Pause")
            if self._is_current_player_ai():
                self._step_ai_move()

    def _undo_move(self):
        if not self.board.move_history:
            return
        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None

        p1_is_human = self.ai_agents[1] is None
        p2_is_human = self.ai_agents[2] is None

        if p1_is_human and not p2_is_human:
            while self.board.move_history and self.board.current_player != 1:
                rec = self.board.move_history[-1]
                edge_id = rec["edge_id"]
                self.board.undo_move()
                self.edge_drawn_by.pop(edge_id, None)
                if self.move_log.size() > 0:
                    self.move_log.delete(tk.END)
            if self.board.move_history:
                rec = self.board.move_history[-1]
                edge_id = rec["edge_id"]
                self.board.undo_move()
                self.edge_drawn_by.pop(edge_id, None)
                if self.move_log.size() > 0:
                    self.move_log.delete(tk.END)
        else:
            rec = self.board.move_history[-1]
            edge_id = rec["edge_id"]
            self.board.undo_move()
            self.edge_drawn_by.pop(edge_id, None)
            if self.move_log.size() > 0:
                self.move_log.delete(tk.END)

        self.last_move = (
            self.board.move_history[-1]["edge_id"] if self.board.move_history else None
        )
        self.suggested_move = None
        self.game_over_card_visible = False
        self.hint_status_lbl.config(text="Move undone", fg=self.TEXT_MUTED)
        self._update_analytics()
        self._draw_board()

    def _show_hint(self):
        if self.board.is_game_over():
            return

        expert = RuleBasedAgent()
        try:
            hint_edge = expert.get_move(self.board.clone())
            hint_edge = self.board.normalize_move(hint_edge)
            self.suggested_move = hint_edge

            edge_tag = self.board.id_to_edge(hint_edge)
            tag_str = f"{edge_tag[0]}({edge_tag[1]},{edge_tag[2]})"

            if self.board.would_complete_box(hint_edge) > 0:
                reason = f"Hint: {tag_str} completes a box (+1)"
            elif not self.board.would_give_away_box(hint_edge):
                reason = f"Hint: {tag_str} is safe"
            else:
                reason = f"Hint: {tag_str} minimal sacrifice"

            self.hint_status_lbl.config(text=reason, fg=self.TEXT_PRIMARY)
            self._draw_board()
        except Exception as e:
            self.hint_status_lbl.config(
                text=f"Hint unavailable: {e}", fg=self.TEXT_MUTED
            )

    # Analytics
    def _update_analytics(self):
        p1_score = self.board.scores[1]
        p2_score = self.board.scores[2]
        total_boxes = self.board.num_boxes

        s1 = SHORT_NAMES.get(self.p1_type, self.p1_type)
        s2 = SHORT_NAMES.get(self.p2_type, self.p2_type)
        self.p1_name_lbl.config(text=f"P1 · {s1}")
        self.p2_name_lbl.config(text=f"P2 · {s2}")
        self.p1_score_val.config(text=str(p1_score))
        self.p2_score_val.config(text=str(p2_score))

        self.progress_canvas.delete("all")
        pw = self.progress_canvas.winfo_width()
        if pw <= 1:
            pw = 230
        ph = 4
        p1_w = (p1_score / total_boxes) * pw
        p2_w = (p2_score / total_boxes) * pw

        if p1_w > 0:
            self.progress_canvas.create_rectangle(
                0, 0, p1_w, ph, fill=self.P1_COLOR, outline=""
            )
        if p2_w > 0:
            self.progress_canvas.create_rectangle(
                pw - p2_w, 0, pw, ph, fill=self.P2_COLOR, outline=""
            )

        captured_boxes = p1_score + p2_score
        self.tactical_boxes_lbl.config(text=f"Boxes: {captured_boxes} / {total_boxes}")
        edges_left = len(self.board.get_valid_moves())
        self.tactical_edges_lbl.config(
            text=f"Edges: {edges_left} / {self.board.total_edges}"
        )

        crit_count = sum(
            1
            for r in range(self.rows)
            for c in range(self.cols)
            if self.board.box_edge_counts[r][c] == 3
        )
        if crit_count > 0:
            self.tactical_critical_lbl.config(
                text=f"Critical (3-sided): {crit_count}", fg=self.P2_COLOR
            )
        else:
            self.tactical_critical_lbl.config(
                text="Critical (3-sided): 0", fg=self.TEXT_MUTED
            )

        if not self.board.is_game_over():
            curr_p = self.board.current_player
            curr_type = s1 if curr_p == 1 else s2
            col = self.P1_COLOR if curr_p == 1 else self.P2_COLOR
            self.turn_pill.config(text=f"Player {curr_p}'s Turn ({curr_type})", fg=col)

    def _handle_game_over(self):
        self.game_over_card_visible = True
        winner = self.board.get_winner()
        p1_score = self.board.scores[1]
        p2_score = self.board.scores[2]

        if winner == 1:
            self.session_scores[1] += 1
            msg = f"Player 1 Wins ({p1_score} - {p2_score})"
            banner_col = self.P1_COLOR
        elif winner == 2:
            self.session_scores[2] += 1
            msg = f"Player 2 Wins ({p2_score} - {p1_score})"
            banner_col = self.P2_COLOR
        else:
            self.session_scores["draw"] += 1
            msg = f"Draw ({p1_score} - {p2_score})"
            banner_col = self.TEXT_SECONDARY

        self.turn_pill.config(text=msg, fg=banner_col)
        self.hint_status_lbl.config(
            text="Match Complete · Press R to play again", fg=self.TEXT_MUTED
        )
        self.btn_play_pause.config(text="New Game")

        self.session_lbl.config(
            text=f"P1: {self.session_scores[1]}  ·  P2: {self.session_scores[2]}  ·  Draws: {self.session_scores['draw']}"
        )
        self.move_log.insert(tk.END, f"— {msg} —")
        self.move_log.see(tk.END)
        self._draw_board()

    # Rendering
    def _draw_board(self):
        self.canvas.delete("all")
        self._compute_geometry()

        x_min, y_min = self._get_dot_pos(0, 0)
        x_max, y_max = self._get_dot_pos(self.rows, self.cols)
        self.canvas.create_rectangle(
            x_min, y_min, x_max, y_max, outline=self.BORDER_HAIRLINE, width=1
        )

        # Boxes
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = self._get_dot_pos(r, c)
                x2, y2 = self._get_dot_pos(r + 1, c + 1)
                owner = self.board.box_owners[r][c]
                edges_count = self.board.box_edge_counts[r][c]

                if owner == 1:
                    self.canvas.create_rectangle(
                        x1 + 2,
                        y1 + 2,
                        x2 - 2,
                        y2 - 2,
                        fill=self.P1_BOX_BG,
                        outline=self.P1_BOX_BORDER,
                        width=1,
                    )
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_text(
                        cx,
                        cy,
                        text="P1",
                        fill=self.P1_COLOR,
                        font=("Helvetica", 10, "bold"),
                    )
                elif owner == 2:
                    self.canvas.create_rectangle(
                        x1 + 2,
                        y1 + 2,
                        x2 - 2,
                        y2 - 2,
                        fill=self.P2_BOX_BG,
                        outline=self.P2_BOX_BORDER,
                        width=1,
                    )
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_text(
                        cx,
                        cy,
                        text="P2",
                        fill=self.P2_COLOR,
                        font=("Helvetica", 10, "bold"),
                    )
                elif edges_count == 3:
                    self.canvas.create_rectangle(
                        x1 + 3,
                        y1 + 3,
                        x2 - 3,
                        y2 - 3,
                        outline=self.P2_COLOR,
                        dash=(3, 3),
                        width=1,
                    )

        # Empty Edges
        for e, (x1, y1, x2, y2) in self.edge_coords.items():
            if (
                not self.board.edges[e]
                and e != self.hovered_edge
                and e != self.suggested_move
            ):
                self.canvas.create_line(
                    x1, y1, x2, y2, fill=self.LINE_EMPTY, width=2, capstyle=tk.ROUND
                )

        # Hint
        if (
            self.suggested_move is not None
            and not self.board.edges[self.suggested_move]
        ):
            x1, y1, x2, y2 = self.edge_coords[self.suggested_move]
            self.canvas.create_line(
                x1, y1, x2, y2, fill=self.LINE_HINT, width=4, capstyle=tk.ROUND
            )

        # Hover
        if self.hovered_edge is not None and not self.board.edges[self.hovered_edge]:
            x1, y1, x2, y2 = self.edge_coords[self.hovered_edge]
            self.canvas.create_line(
                x1, y1, x2, y2, fill=self.LINE_HOVER, width=3, capstyle=tk.ROUND
            )

        # Drawn Edges
        for e, (x1, y1, x2, y2) in self.edge_coords.items():
            if self.board.edges[e]:
                drawer = self.edge_drawn_by.get(e, 1)
                color = self.P1_COLOR if drawer == 1 else self.P2_COLOR

                if e == self.last_move:
                    self.canvas.create_line(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill=self.LINE_LAST_MOVE,
                        width=6,
                        capstyle=tk.ROUND,
                    )

                self.canvas.create_line(
                    x1, y1, x2, y2, fill=color, width=3.5, capstyle=tk.ROUND
                )

        # Dots
        dot_r = 5
        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                x, y = self._get_dot_pos(r, c)
                self.canvas.create_oval(
                    x - dot_r - 1,
                    y - dot_r - 1,
                    x + dot_r + 1,
                    y + dot_r + 1,
                    fill=self.DOT_OUTLINE,
                    outline="",
                )
                self.canvas.create_oval(
                    x - dot_r,
                    y - dot_r,
                    x + dot_r,
                    y + dot_r,
                    fill=self.DOT_COLOR,
                    outline="",
                )

        if self.board.is_game_over() and self.game_over_card_visible:
            self._draw_victory_card()

    def _draw_victory_card(self):
        cx = self.canvas_width / 2
        cy = self.canvas_height / 2
        w, h = 300, 130

        self.canvas.create_rectangle(
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            fill="#121214",
            outline=self.BORDER_HAIRLINE,
            width=1,
        )

        winner = self.board.get_winner()
        p1_score = self.board.scores[1]
        p2_score = self.board.scores[2]

        if winner == 1:
            title_text = "PLAYER 1 WINS"
            title_col = self.P1_COLOR
            sub_text = f"{p1_score} – {p2_score}"
        elif winner == 2:
            title_text = "PLAYER 2 WINS"
            title_col = self.P2_COLOR
            sub_text = f"{p2_score} – {p1_score}"
        else:
            title_text = "DRAW"
            title_col = self.TEXT_SECONDARY
            sub_text = f"{p1_score} – {p2_score}"

        self.canvas.create_text(
            cx, cy - 28, text=title_text, font=("Helvetica", 11, "bold"), fill=title_col
        )
        self.canvas.create_text(
            cx, cy - 6, text=sub_text, font=("Helvetica", 11), fill=self.TEXT_SECONDARY
        )

        self.canvas.create_rectangle(
            cx - 60, cy + 16, cx + 60, cy + 42, fill=self.TEXT_PRIMARY, outline=""
        )
        self.canvas.create_text(
            cx,
            cy + 29,
            text="Play Again",
            font=("Helvetica", 9, "bold"),
            fill=self.BG_WINDOW,
        )


def main():
    root = tk.Tk()
    app = DotsAndBoxesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
