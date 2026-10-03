from __future__ import annotations

"""
Single-page web app: upload a screenshot, get the animated solution.

Run:
    venv\\Scripts\\python app.py
then open http://localhost:8000
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from image_to_state import bytes_to_state
from solver import from_screenshot_state, solve
from visualize import build_html

ROOT = Path(__file__).parent
# Files the page loads; the solver itself runs in the browser (worker.js).
STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "worker.js": "text/javascript; charset=utf-8",
    "app.py": "text/plain; charset=utf-8",
    "image_to_state.py": "text/plain; charset=utf-8",
    "solver.py": "text/plain; charset=utf-8",
    "visualize.py": "text/plain; charset=utf-8",
    "visualizer_template.html": "text/html; charset=utf-8",
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def solve_screenshot(data: bytes) -> dict:
    """Parse and solve an encoded screenshot; return the JSON response body."""
    state, palette, _, _ = bytes_to_state(data)

    started = time.perf_counter()
    solution = solve(from_screenshot_state(state))
    elapsed = time.perf_counter() - started

    if solution is None:
        return {"error": "The detected board has no solution. Check the screenshot."}

    return {
        "state": state,
        "moves": len(solution.moves),
        "expanded": solution.expanded_states,
        "seconds": round(elapsed, 2),
        "html": build_html(solution, palette),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        name = self.path.lstrip("/") or "index.html"
        if name not in STATIC_FILES:
            self.send_error(404)
            return
        self._send(200, (ROOT / name).read_bytes(), STATIC_FILES[name])

    def do_POST(self) -> None:
        if self.path != "/solve":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_UPLOAD_BYTES:
            self._send_json(400, {"error": "Upload an image of at most 20 MB."})
            return

        try:
            body = solve_screenshot(self.rfile.read(length))
        except (RuntimeError, ValueError) as exc:
            # Parser/solver failures carry a readable message.
            self._send_json(422, {"error": str(exc)})
            return

        self._send_json(422 if "error" in body else 200, body)

    def _send_json(self, status: int, body: dict) -> None:
        self._send(status, json.dumps(body).encode("utf-8"), "application/json")

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color Puzzle solver web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
