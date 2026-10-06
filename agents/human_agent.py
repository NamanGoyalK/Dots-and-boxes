"""
Human Agent - Allows manual move input in console.
"""

from typing import Union, Tuple
from core.base_agent import BaseAgent
from core.board import Board


class HumanAgent(BaseAgent):
    """Prompts human user in terminal to enter an edge ID or coordinate."""

    def __init__(self, name: str = "Human"):
        super().__init__(name=name)

    def get_move(self, board: Board) -> Union[int, Tuple[str, int, int]]:
        valid_moves = board.get_valid_moves()
        print("\n" + board.render_str(show_indices=True))
        print(f"Valid edge IDs: {valid_moves}")
        
        while True:
            try:
                user_input = input(f"[{self.name}] Enter move (edge ID or 'H r c' / 'V r c'): ").strip()
                if not user_input:
                    continue
                    
                parts = user_input.split()
                if len(parts) == 1:
                    edge_id = int(parts[0])
                    if edge_id in valid_moves:
                        return edge_id
                    print(f"Edge {edge_id} is not valid. Choose from: {valid_moves}")
                elif len(parts) == 3:
                    etype = parts[0].upper()
                    r, c = int(parts[1]), int(parts[2])
                    edge_id = board.edge_to_id(etype, r, c)
                    if edge_id in valid_moves:
                        return edge_id
                    print(f"Move ({etype}, {r}, {c}) is already taken or invalid.")
                else:
                    print("Invalid input format. Use an integer ID like '5' or 'H 0 1'.")
            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Try again.")
