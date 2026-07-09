# Run this as a cell in your Colab notebook, AFTER Part C has already run once
# (needs clean_df, X, y, groups, and the catalog already loaded in memory).
# Produces: dabc_data.json — every catalogued DABC, tested ones marked "tested" with
# real severity, untested ones marked "predicted" with the model's prediction + confidence.

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---- 1. Re-derive the same features for the FULL catalog (not just the 13 tested DABCs) ----
full_catalog = catalog.copy()  # the 88-row catalog from Part B, already has module/function/is_class

def module_depth(module):
    return str(module).count(".") + 1

def is_private_module(module):
    return any(part.startswith("_") for part in str(module).split("."))

def infer_task_type(function_name):
    name = str(function_name)
    if "Classifier" in name:
        return "classifier"
    elif "Regressor" in name:
        return "regressor"
    elif "Cluster" in name or "KMeans" in name:
        return "clusterer"
    elif "Transformer" in name:
        return "transformer"
    else:
        return "other"

ENSEMBLE_PATTERNS = ["RandomForest", "ExtraTrees", "GradientBoosting", "AdaBoost", "Bagging", "Voting", "Stacking", "HistGradientBoosting"]
def is_ensemble_method(function_name):
    name = str(function_name)
    return any(p in name for p in ENSEMBLE_PATTERNS)

full_catalog["has_parameter"] = full_catalog["parameter"].notna()
full_catalog["module_depth"] = full_catalog["module"].apply(module_depth)
full_catalog["is_private_module"] = full_catalog["module"].apply(is_private_module)
full_catalog["task_type"] = full_catalog["function"].apply(infer_task_type)
full_catalog["is_ensemble"] = full_catalog["function"].apply(is_ensemble_method)

feature_cols = ["category", "library", "has_parameter", "module_depth", "is_private_module", "task_type", "is_ensemble"]
# harness_category may not exist for rows that were never classified — guard for that
if "harness_category" not in full_catalog.columns:
    full_catalog["harness_category"] = "unknown"
full_catalog["category"] = full_catalog["harness_category"]

X_full = pd.get_dummies(full_catalog[feature_cols], columns=["category", "library", "task_type"])
# align columns to whatever the trained model was fit on (X from Part C)
X_full = X_full.reindex(columns=X.columns, fill_value=False)

# ---- 2. Train the FINAL model on ALL tested data (no CV split — this is for deployment, not evaluation) ----
final_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
final_model.fit(X, y)

# ---- 3. Predict for every DABC in the full catalog ----
pred_labels = final_model.predict(X_full)
pred_proba = final_model.predict_proba(X_full)
risky_idx = list(final_model.classes_).index("risky")

# ---- 4. Build ground-truth lookup for the 13 tested DABCs (majority label across seeds) ----
tested_severity = (
    clean_df.groupby("dabc_index")["severity_binary"]
    .agg(lambda s: s.value_counts().idxmax())
    .to_dict()
)
tested_confidence = (
    clean_df.groupby("dabc_index")["severity_binary"]
    .agg(lambda s: s.value_counts().max() / len(s))
    .to_dict()
)

# ---- 5. Assemble the final record for each DABC ----
records = []
for i, (idx, row) in enumerate(full_catalog.iterrows()):
    is_tested = idx in tested_severity
    record = {
        "id": int(idx),
        "library": row.get("library", ""),
        "module": row.get("module", ""),
        "function": row.get("function", ""),
        "parameter": row.get("parameter") if pd.notna(row.get("parameter")) else None,
        "old_version": row.get("old_version") if pd.notna(row.get("old_version")) else None,
        "new_version": row.get("new_version") if pd.notna(row.get("new_version")) else None,
        "dabc_msg": row.get("dabc_msg") if pd.notna(row.get("dabc_msg")) else None,
        "task_type": row.get("task_type", "other"),
        "is_ensemble": bool(row.get("is_ensemble", False)),
        "severity": tested_severity[idx] if is_tested else pred_labels[i],
        "source": "tested" if is_tested else "predicted",
        "confidence": round(float(tested_confidence[idx]), 3) if is_tested else round(float(pred_proba[i][risky_idx] if pred_labels[i] == "risky" else 1 - pred_proba[i][risky_idx]), 3),
    }
    records.append(record)

# ---- 6. Save and download ----
output_path = "/content/dabc_data.json"
with open(output_path, "w") as f:
    json.dump({
        "generated_from": "SilentDrift severity model v1",
        "total_dabcs": len(records),
        "tested_count": sum(1 for r in records if r["source"] == "tested"),
        "predicted_count": sum(1 for r in records if r["source"] == "predicted"),
        "dabcs": records,
    }, f, indent=2)

print(f"Saved {len(records)} DABCs to {output_path}")
print(f"  Tested (real differential-testing results): {sum(1 for r in records if r['source']=='tested')}")
print(f"  Predicted (model-inferred, no execution): {sum(1 for r in records if r['source']=='predicted')}")

from google.colab import files
files.download(output_path)
