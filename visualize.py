from __future__ import annotations

"""
Render a solved puzzle as a self-contained, animated HTML page.

The page is built from visualizer_template.html with the palette, every
intermediate state and the move list embedded as JSON, so it opens
directly from disk without a web server.
"""

import json
from pathlib import Path
from typing import Sequence

from solver import Solution

TEMPLATE_PATH = Path(__file__).with_name("visualizer_template.html")
PLACEHOLDER = "/*__PUZZLE_DATA__*/"


def export_html(
    solution: Solution,
    palette: Sequence[Sequence[float]],
    output_path: str | Path = "solution.html",
    capacity: int = 4,
) -> Path:
    """Write the solution viewer and return its path."""
    output_path = Path(output_path)
    output_path.write_text(build_html(solution, palette, capacity), encoding="utf-8")
    return output_path


def build_html(
    solution: Solution,
    palette: Sequence[Sequence[float]],
    capacity: int = 4,
) -> str:
    """Return the solution viewer page as a string."""
    data = {
        "capacity": capacity,
        "palette": [[round(float(v)) for v in rgb] for rgb in palette],
        "states": [[list(tube) for tube in state] for state in solution.states],
        "moves": [
            {
                "source": move.source,
                "destination": move.destination,
                "amount": move.amount,
                # The moved color is the destination's new bottom block.
                "color": after[move.destination][0],
            }
            for move, after in zip(solution.moves, solution.states[1:])
        ],
    }

    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    return html.replace(PLACEHOLDER, json.dumps(data))
