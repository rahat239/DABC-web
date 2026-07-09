# SilentDrift

A web app that tells developers whether upgrading scikit-learn/pandas/numpy will silently change their results — before they upgrade.

## What's in this repo

- `app/index.html` — the whole web app. Single file, no build step, no backend. Currently loaded with **sample data** (13 DABCs) so it's demoable immediately.
- `app/dabc_data.json` — the data the app reads. Replace this with your real export (see below) to go from 13 sample DABCs to your full real catalog.
- `export/export_for_webapp.py` — run this in your Colab session (after Part C has already run) to generate a real `dabc_data.json` from your actual trained model and catalog.

## Getting your real data into the app

1. Open your `DABC_pipeline_all_in_one.ipynb` in Colab and run it through Part C, section 7 (SHAP), so `clean_df`, `X`, `y`, `groups`, and `catalog` are all populated in memory.
2. Add a new cell at the end, paste in the contents of `export/export_for_webapp.py`, and run it.
3. It'll download `dabc_data.json` to your computer automatically.
4. Replace `app/dabc_data.json` with the downloaded file (same filename, same location).
5. Open `app/index.html` in a browser — you should now see all your catalogued DABCs, not just the 13 sample ones. Tested DABCs (the ones you differentially tested) show a `✓ tested` tag; the rest show `◐ predicted · N%` — the model's confidence for DABCs it never actually executed.

## Running it locally

No build step needed. Either:
- Open `app/index.html` directly in a browser, or
- `cd app && python3 -m http.server 8000` and visit `http://localhost:8000` (needed if your browser blocks `fetch()` on `file://` URLs — Chrome does this by default)

## Deploying

**GitHub Pages** (matches your existing pattern from the portfolio/PyPIGuard splash pages):
1. Push the `app/` folder's contents to a repo (or a `gh-pages` branch / `docs/` folder).
2. Enable GitHub Pages in the repo settings, pointing at that folder.
3. Done — no server, no cold starts, no Render free-tier sleep issues.

## Honest scope note

The 13 "tested" DABCs are the ones your differential-testing pipeline actually executed successfully. The remaining ~75 catalogued DABCs are shown with model-predicted severity — useful signal, but not verified the same way. If you differentially test more DABCs later (e.g. by recovering `NO_COMPATIBLE_PYTHON` coverage), just re-run the export script and replace the JSON — the app itself doesn't need to change.
