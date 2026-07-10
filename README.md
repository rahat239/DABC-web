# SilentDrift

A real web app — Flask backend + trained model, not a static lookup table — that tells developers whether upgrading scikit-learn/pandas/numpy will silently change their results, before they upgrade.

## Three real capabilities

1. **Browse the catalog** — all 88 known DABCs, each marked `tested` (real differential-testing result) or `predicted` (model inference, no execution).
2. **Live prediction** — check *any* function, not just the 88 catalogued ones. The trained model runs in real time.
3. **Code scanning** — paste actual Python code; the backend parses it with Python's `ast` module, finds every call to a catalogued function, and checks whether the risky default was left unset (i.e. your code is actually exposed).

## What's in this repo

```
backend/
  app.py               — Flask app: serves the frontend + 3 API endpoints
  features.py           — feature engineering, MUST match what trained the model
  requirements.txt
  dabc_data.json         — your real 88-DABC catalog
  model.joblib            — your REAL trained model (sklearn 1.6.1, RandomForest, 11 features)
app/
  index.html             — the frontend (served by Flask, not standalone anymore)
export/
  export_for_webapp.py   — run in Colab again if you regenerate the catalog/model
```

## Status: real data, real model, tested end-to-end

`backend/model.joblib` and `backend/dabc_data.json` are your actual Colab outputs — not placeholders. Verified before shipping: loads cleanly with matching scikit-learn 1.6.1 (no version warnings), runs correctly under both Flask's dev server and gunicorn, and its live predictions match your real findings exactly (`RandomForestClassifier` → safe, 96.1% confidence; `ExtraTreeClassifier` → risky, 100% confidence — the ensemble-averaging pattern from your SHAP analysis, reproduced live).

**One bug fixed during testing:** `/api/predict` originally ran a fresh live prediction even for functions already in the catalog, which could give a confusing, lower-confidence answer for something already verified (e.g. `ExtraTreesClassifier` showed 57% live vs. its real 89% tested confidence). Fixed: the endpoint now checks the catalog first and returns the real, verified result(s) when a function is already known — `ExtraTreesClassifier` correctly shows it actually has *two* DABCs affecting it (`n_estimators` and `max_features` changes). Live prediction only kicks in for genuinely new, uncatalogued functions.

## Regenerating the model later

If you differentially test more DABCs (e.g. recovering `NO_COMPATIBLE_PYTHON` coverage) and want to retrain:

1. Open your Colab notebook, run through Part C (through the SHAP/permutation importance cells) so `clean_df`, `X`, `y`, `catalog` are in memory.
2. Paste `export/export_for_webapp.py` into a new cell and run it. It now downloads two files: `dabc_data.json` and `model.joblib`.
3. Replace both files in `backend/` with the downloaded ones.
4. Restart the server locally to confirm: `cd backend && python3 app.py` — check the startup log says `Loaded model with N features` with no warnings.

## Running locally

```
cd backend
pip install -r requirements.txt
python3 app.py
```

Visit `http://localhost:5000`. Or with gunicorn (matches what Render runs in production):

```
gunicorn --bind 127.0.0.1:5000 app:app
```

## Deploying on Render

This needs a Web Service, not a Static Site — it has a real backend now.

1. New -> Web Service, connect the DABC-web repo.
2. Root Directory: `backend`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn --bind 0.0.0.0:$PORT app:app`
5. Deploy.

Render's free tier still sleeps after inactivity (same cold-start behavior you've hit with your other projects) — the GitHub Pages splash-page trick you used for PyPIGuard/job-fraud would work the same way here if you want to avoid that.

## API reference

**GET /api/catalog** -> the full catalog, same shape as before.

**POST /api/predict**
```
{ "function": "GradientBoostingRegressor", "library": "scikit-learn", "module": "sklearn.ensemble._gb" }
```
returns `{ "severity": "risky", "confidence": 0.99, "reasons": [...], "note": "..." }`

**POST /api/scan**
```
{ "code": "from sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier()\n" }
```
returns `{ "total_findings": N, "risky_findings": N, "findings": [...] }`

## Honest scope note

`/api/predict` extrapolates the model beyond the 13 DABCs it was actually trained on — treat it as a signal to investigate, not a verified result (the app says this explicitly in every response). `/api/scan`'s parameter-override detection only works when a DABC's specific parameter was successfully identified during data extraction; some catalog entries have `parameter: null` (message-parsing didn't isolate a single parameter name) and get flagged unconditionally as a conservative fallback — this is documented, expected behavior, not a bug.
