# Practice raceline sketcher

Open [`raceline_sketcher.html`](raceline_sketcher.html) in a browser. It runs locally in the browser and reads files only when you choose them.

Load these three files from the workspace:

- `f1tenth_planning/maps/autodrive_practice_20260924_b.pgm`
- `f1tenth_planning/maps/autodrive_practice_20260924_b.yaml`
- A starting raceline CSV, such as `f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_v300_optimized_mpc_amcl.csv`

Click **Add shaping point**, then click the map where you want the path to move. The pink line is the editable candidate; the cyan source is just a visual reference. Drag points to refine the shape. The influence-radius slider changes how broadly each point blends into the path. Scroll to zoom, middle/right-drag to pan, `Ctrl+Z` to undo, and `Esc` to exit add mode. Arrow keys move a selected point by 2 cm; `Shift`+arrow moves it by 10 cm.

The preview checks the body rectangle expanded by 5 cm against free map cells; unknown cells count as blocked. This is a map/footprint check, not a physics-feasibility or collision guarantee. Export creates a **new** CSV and does not overwrite the input. It recalculates path distance, heading, curvature, and map ray distances, but copies the input speed and acceleration columns. Recompute those in the optimizer and independently validate any exported candidate before use.
