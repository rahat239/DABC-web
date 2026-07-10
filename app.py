"""
SilentDrift backend.

Three real capabilities, not a static lookup:
  GET  /api/catalog        -> the known 88-DABC catalog (tested + predicted)
  POST /api/predict        -> live severity prediction for ANY function, not just the catalog
  POST /api/scan           -> parse pasted Python code, flag every catalogued DABC it's exposed to

Requires model.joblib (trained model + feature schema) and dabc_data.json to be present
in this directory — both come from export_for_webapp.py, run in your Colab session.
"""

import ast
import json
import os

import joblib
from flask import Flask, jsonify, request, send_from_directory

import features

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# CORS: without this, any request from a different origin (e.g. a GitHub Pages
# splash page polling this API while the Render service wakes up) gets its
# response silently blocked by the browser, even though the server responds
# 200 OK. This is not optional for a splash-page-based wake-up flow.
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/api/predict", methods=["OPTIONS"])
@app.route("/api/scan", methods=["OPTIONS"])
def cors_preflight():
    # Browsers send an OPTIONS preflight before cross-origin POST requests with
    # a JSON content-type. Without an explicit 200 response here, the actual
    # POST never gets sent at all.
    return ("", 200)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = BASE_DIR

# ---------------------------------------------------------------------------
# Load model + catalog once at startup, not per-request
# ---------------------------------------------------------------------------
MODEL = None
FEATURE_COLUMNS = None
CATALOG = []

model_path = os.path.join(BASE_DIR, "model.joblib")
catalog_path = os.path.join(BASE_DIR, "dabc_data.json")

if os.path.exists(model_path):
    _bundle = joblib.load(model_path)
    MODEL = _bundle["model"]
    FEATURE_COLUMNS = _bundle["feature_columns"]
    print(f"Loaded model with {len(FEATURE_COLUMNS)} features.")
else:
    print(f"WARNING: {model_path} not found. /api/predict and live-inference parts of "
          f"/api/scan will return an error until you add it (see README).")

if os.path.exists(catalog_path):
    with open(catalog_path) as f:
        CATALOG = json.load(f)["dabcs"]
    print(f"Loaded catalog with {len(CATALOG)} DABCs.")
else:
    print(f"WARNING: {catalog_path} not found. /api/catalog and /api/scan will return empty results.")

# Index catalog by function name for fast scan lookups (case-sensitive, matches source data)
CATALOG_BY_FUNCTION = {}
for entry in CATALOG:
    CATALOG_BY_FUNCTION.setdefault(entry["function"], []).append(entry)


# ---------------------------------------------------------------------------
# Serve the frontend itself (single Render service, no separate static host needed)
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ---------------------------------------------------------------------------
# GET /api/catalog
# ---------------------------------------------------------------------------
@app.route("/api/catalog")
def get_catalog():
    return jsonify({
        "total_dabcs": len(CATALOG),
        "tested_count": sum(1 for d in CATALOG if d["source"] == "tested"),
        "predicted_count": sum(1 for d in CATALOG if d["source"] == "predicted"),
        "dabcs": CATALOG,
    })


# ---------------------------------------------------------------------------
# POST /api/predict
# body: { "function": "SomeClassifier", "library": "scikit-learn",
#         "module": "sklearn.some.module" (optional), "parameter": "n_jobs" (optional) }
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded on this server. See README for setup."}), 503

    data = request.get_json(force=True, silent=True) or {}
    function_name = data.get("function", "").strip()
    library = data.get("library", "scikit-learn").strip()
    module = data.get("module", "").strip()
    parameter = data.get("parameter", "").strip()

    if not function_name:
        return jsonify({"error": "'function' is required."}), 400

    # If this function is already in the catalog, return the real (tested or previously-
    # predicted) result instead of a fresh live prediction — avoids giving a different,
    # potentially lower-confidence answer for something already verified.
    existing = CATALOG_BY_FUNCTION.get(function_name, [])
    if existing:
        return jsonify({
            "function": function_name,
            "library": library or existing[0]["library"],
            "already_in_catalog": True,
            "matches": [
                {
                    "severity": e["severity"],
                    "source": e["source"],
                    "confidence": e["confidence"],
                    "parameter": e.get("parameter"),
                    "old_version": e.get("old_version"),
                    "new_version": e.get("new_version"),
                    "dabc_msg": e.get("dabc_msg"),
                }
                for e in existing
            ],
            "note": f"'{function_name}' already has {len(existing)} known DABC(s) in the catalog "
                    f"— showing those verified/previously-predicted results rather than a fresh "
                    f"live prediction, to avoid a confusing discrepancy.",
        })

    task_type = features.infer_task_type(function_name)
    is_ensemble = features.is_ensemble_method(function_name)
    category = features.infer_category(library, function_name)
    mod_depth = features.module_depth(module) if module else 2
    is_private = features.is_private_module(module) if module else False
    has_param = bool(parameter)

    row = features.build_feature_row(
        FEATURE_COLUMNS,
        category=category, library=library, has_parameter=has_param,
        module_depth_val=mod_depth, is_private_module_val=is_private,
        task_type=task_type, is_ensemble=is_ensemble,
    )

    import pandas as pd
    X_row = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred = MODEL.predict(X_row)[0]
    proba = MODEL.predict_proba(X_row)[0]
    risky_idx = list(MODEL.classes_).index("risky")
    confidence = float(proba[risky_idx] if pred == "risky" else 1 - proba[risky_idx])

    reasons = []
    if is_ensemble:
        reasons.append("Ensemble method — averaging across estimators tends to dilute the effect of a single default change.")
    else:
        reasons.append("Not an ensemble method — more directly exposed to a single default change.")
    if task_type == "classifier":
        reasons.append("Classification task — historically more stable to the default changes seen in this dataset.")
    elif task_type == "regressor":
        reasons.append("Regression task — historically more sensitive to the default changes seen in this dataset.")

    return jsonify({
        "function": function_name,
        "library": library,
        "severity": pred,
        "confidence": round(confidence, 3),
        "task_type": task_type,
        "is_ensemble": is_ensemble,
        "reasons": reasons,
        "note": "Live prediction — this function/parameter combination was not necessarily "
                "part of the 88-DABC catalog. Model trained on 13 differentially-tested DABCs; "
                "treat this as a signal to investigate further, not a verified result.",
    })


# ---------------------------------------------------------------------------
# POST /api/scan
# body: { "code": "<pasted python source>" }
# Finds every call to a catalogued function and checks whether the risky default
# parameter was left unset (i.e. the code is actually exposed to that DABC).
# ---------------------------------------------------------------------------
def extract_call_name(call_node):
    """Get the plain function/class name from an ast.Call, handling both
    bare calls (RandomForestClassifier(...)) and attribute calls (sklearn.ensemble.RandomForestClassifier(...))."""
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def extract_keyword_names(call_node):
    return {kw.arg for kw in call_node.keywords if kw.arg is not None}


@app.route("/api/scan", methods=["POST"])
def scan_code():
    data = request.get_json(force=True, silent=True) or {}
    code = data.get("code", "")

    if not code.strip():
        return jsonify({"error": "'code' is required and cannot be empty."}), 400

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return jsonify({"error": f"Could not parse code: {e}"}), 400

    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = extract_call_name(node)
        if not call_name or call_name not in CATALOG_BY_FUNCTION:
            continue

        used_kwargs = extract_keyword_names(node)
        line_no = getattr(node, "lineno", None)

        for dabc in CATALOG_BY_FUNCTION[call_name]:
            param = dabc.get("parameter")
            # If the DABC has a specific named parameter, only flag if the code
            # did NOT explicitly set it (i.e. it's actually relying on the default).
            # If no specific parameter is identified (message-level DABC), always flag —
            # we can't verify safety without knowing exactly which value it needs.
            exposed = (param is None) or (param not in used_kwargs)
            if exposed:
                findings.append({
                    "line": line_no,
                    "function": call_name,
                    "severity": dabc["severity"],
                    "source": dabc["source"],
                    "confidence": dabc.get("confidence"),
                    "parameter": param,
                    "dabc_msg": dabc.get("dabc_msg"),
                    "old_version": dabc.get("old_version"),
                    "new_version": dabc.get("new_version"),
                    "reason": (
                        f"relies on the default value of `{param}`" if param
                        else "affected by a version-specific default change with no single identified parameter"
                    ),
                })

    findings.sort(key=lambda f: (f["severity"] != "risky", f["line"] or 0))

    return jsonify({
        "total_findings": len(findings),
        "risky_findings": sum(1 for f in findings if f["severity"] == "risky"),
        "findings": findings,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)