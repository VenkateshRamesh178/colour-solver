// Runs the Python solver in the browser via Pyodide, so the site can be
// hosted as static files (e.g. GitHub Pages) with no server.
//
// Messages in:  { id, bytes: Uint8Array }   an encoded screenshot
// Messages out: { type: "ready" }
//               { type: "result", id, body } same JSON body app.py returns
//               { type: "error", id?, message }

// Must be a module worker: Pyodide 314's classic pyodide.js fails under
// importScripts() in Chrome.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const SOURCES = [
  "app.py",
  "image_to_state.py",
  "solver.py",
  "visualize.py",
  "visualizer_template.html",
];
const WORKDIR = "/home/pyodide/solver";

async function load() {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "opencv-python", "scikit-learn"]);

  pyodide.FS.mkdirTree(WORKDIR);
  await Promise.all(SOURCES.map(async name => {
    // Revalidate so a redeploy is picked up despite Pages' 10-minute cache.
    const res = await fetch(name, { cache: "no-cache" });
    if (!res.ok) throw new Error(`Could not load ${name} (${res.status}).`);
    pyodide.FS.writeFile(`${WORKDIR}/${name}`, await res.text());
  }));

  pyodide.runPython(`
import json, sys
sys.path.insert(0, "${WORKDIR}")
from app import solve_screenshot

def run(data):
    try:
        return json.dumps(solve_screenshot(bytes(data.to_py())))
    except (RuntimeError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
`);
  return pyodide.globals.get("run");
}

const ready = load();
ready.then(
  () => postMessage({ type: "ready" }),
  err => postMessage({ type: "error", message: `Failed to load the solver: ${err.message}` }),
);

onmessage = async ({ data: { id, bytes } }) => {
  try {
    const run = await ready;
    postMessage({ type: "result", id, body: JSON.parse(run(bytes)) });
  } catch (err) {
    postMessage({ type: "error", id, message: err.message });
  }
};
