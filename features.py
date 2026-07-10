"""
Feature engineering for DABC severity prediction.
MUST stay identical to what was used in Colab to train model.joblib — if these functions
drift from the training-time versions, live predictions will be silently wrong.
"""

ENSEMBLE_PATTERNS = ["RandomForest", "ExtraTrees", "GradientBoosting", "AdaBoost",
                     "Bagging", "Voting", "Stacking", "HistGradientBoosting"]


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


def is_ensemble_method(function_name):
    name = str(function_name)
    return any(p in name for p in ENSEMBLE_PATTERNS)


def module_depth(module):
    return str(module).count(".") + 1 if module else 1


def is_private_module(module):
    if not module:
        return False
    return any(part.startswith("_") for part in str(module).split("."))


def infer_category(library, function_name, is_class_hint=None):
    """For arbitrary user-submitted functions not in the trained catalog, infer which
    harness category they'd fall into. is_class_hint lets a caller override the
    CamelCase-based guess if they know better (e.g. from an AST-parsed class definition)."""
    if is_class_hint is None:
        is_class_hint = bool(function_name) and function_name[0].isupper()

    lib = str(library).lower()
    if "pandas" in lib:
        return "pandas_method"
    if "numpy" in lib:
        return "numpy_function"
    return "sklearn_estimator" if is_class_hint else "sklearn_function"


def build_feature_row(feature_columns, *, category, library, has_parameter,
                       module_depth_val, is_private_module_val, task_type, is_ensemble):
    """Build a single feature row aligned to the exact columns the trained model expects.
    Unknown/unseen categorical values simply produce an all-False one-hot row for that
    family, which is the correct behavior (the model just sees 'none of the known values')."""
    row = {col: False for col in feature_columns}

    if "has_parameter" in row:
        row["has_parameter"] = bool(has_parameter)
    if "module_depth" in row:
        row["module_depth"] = module_depth_val
    if "is_private_module" in row:
        row["is_private_module"] = bool(is_private_module_val)
    if "is_ensemble" in row:
        row["is_ensemble"] = bool(is_ensemble)

    cat_col = f"category_{category}"
    if cat_col in row:
        row[cat_col] = True

    lib_col = f"library_{library}"
    if lib_col in row:
        row[lib_col] = True

    task_col = f"task_type_{task_type}"
    if task_col in row:
        row[task_col] = True

    return row
