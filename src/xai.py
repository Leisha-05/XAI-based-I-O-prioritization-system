"""
src/xai.py — Explainability for the ML scheduler

Provides:
  - Feature contribution approximation via tree path walking
  - Human-readable explanation generation
  - SHAP-style waterfall data for the frontend chart
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Any


# ── Tree-path feature contributions ──────────────────────────

def _walk_tree(tree, x_row: np.ndarray, lr: float = 1.0) -> np.ndarray:
    """
    Walk a single sklearn decision tree for one sample.
    Returns per-feature contribution array.
    """
    n_features = x_row.shape[0]
    contribs   = np.zeros(n_features)

    node = 0
    prev_val = tree.value[0, 0, 0]

    while tree.feature[node] >= 0:          # not a leaf
        feat = tree.feature[node]
        child = (tree.children_right[node]
                 if x_row[feat] > tree.threshold[node]
                 else tree.children_left[node])
        curr_val = tree.value[child, 0, 0]
        contribs[feat] += lr * (curr_val - prev_val)
        prev_val = curr_val
        node = child

    return contribs


def compute_contributions(model, feature_cols: List[str],
                           X: pd.DataFrame) -> np.ndarray:
    """
    Compute approximate SHAP values for a GradientBoostingRegressor
    by summing each tree's path contributions.

    Returns ndarray shape (n_samples, n_features).
    """
    X_arr    = X.values.astype(np.float32)
    n_s, n_f = X_arr.shape
    total    = np.zeros((n_s, n_f))

    for tree_wrapper in model.estimators_.ravel():
        tree = tree_wrapper.tree_
        for i in range(n_s):
            total[i] += _walk_tree(tree, X_arr[i], model.learning_rate)

    return total


# ── Human-readable explanation ────────────────────────────────

_PRIORITY_LABEL = {0: "Low", 1: "Medium", 2: "High"}
_TYPE_LABEL     = {0: "Read", 1: "Write"}

def _describe(feature: str, value: float) -> str:
    if feature == "priority":
        return f"{_PRIORITY_LABEL.get(int(value), '?')} priority"
    if feature == "type":
        return f"{_TYPE_LABEL.get(int(value), '?')} request"
    if feature == "cylinder":
        return f"cylinder {int(value)}"
    if feature == "seek_distance":
        if value < 20:
            return f"short seek ({int(value)} tracks)"
        elif value < 80:
            return f"medium seek ({int(value)} tracks)"
        else:
            return f"long seek ({int(value)} tracks)"
    if feature == "deadline":
        if value < 50:
            return f"urgent deadline ({int(value)} ms)"
        elif value < 200:
            return f"moderate deadline ({int(value)} ms)"
        else:
            return f"relaxed deadline ({int(value)} ms)"
    if feature == "size":
        if value < 64:
            return f"small I/O ({int(value)} KB)"
        elif value < 256:
            return f"medium I/O ({int(value)} KB)"
        else:
            return f"large I/O ({int(value)} KB)"
    return f"{feature}={value:.1f}"


def generate_explanation(request: dict, contributions: np.ndarray,
                          feature_cols: List[str], top_n: int = 3) -> Dict[str, Any]:
    """
    Build a human-readable explanation for one scheduling decision.

    Args:
        request      : dict with all feature values + id/score
        contributions: 1-D array of per-feature contributions
        feature_cols : ordered list of feature names
        top_n        : how many top drivers to surface

    Returns dict with:
        summary      : one-sentence explanation
        drivers      : list of {feature, value, contribution, description}
        waterfall    : data for frontend SHAP waterfall bar chart
    """
    contribs_list = [
        {
            "feature":      f,
            "value":        float(request.get(f, 0)),
            "contribution": float(c),
            "description":  _describe(f, request.get(f, 0)),
        }
        for f, c in zip(feature_cols, contributions)
    ]
    contribs_list.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top = contribs_list[:top_n]

    positive = [d for d in top if d["contribution"] >= 0]
    negative = [d for d in top if d["contribution"] < 0]

    if positive:
        pos_str = " and ".join(d["description"] for d in positive[:2])
        if negative:
            neg_str = negative[0]["description"]
            summary = f"Prioritised due to {pos_str}, partly offset by {neg_str}."
        else:
            summary = f"Prioritised due to {pos_str}."
    else:
        summary = "Deprioritised — all contributing factors score negatively."

    # Waterfall: baseline + each feature contribution bar
    baseline  = float(np.mean(contributions)) if len(contributions) > 0 else 0.0
    waterfall = []
    running   = baseline
    for d in contribs_list:
        waterfall.append({
            "feature":      d["feature"],
            "description":  d["description"],
            "contribution": d["contribution"],
            "running_total": round(running + d["contribution"], 5),
        })
        running += d["contribution"]

    return {
        "summary":   summary,
        "drivers":   contribs_list,
        "waterfall": waterfall,
        "score":     round(float(request.get("ml_score", 0)), 4),
    }


# ── Bulk XAI for ML scheduler run ────────────────────────────

def explain_schedule(model, feature_cols: List[str],
                     served_order: List[dict]) -> List[dict]:
    """
    Attach explanation to every request in a served schedule.
    served_order: list of request dicts that include all feature values.
    Returns same list with 'explanation' key added.
    """
    if not served_order:
        return served_order

    X = pd.DataFrame([{f: r[f] for f in feature_cols} for r in served_order])
    all_contribs = compute_contributions(model, feature_cols, X)

    result = []
    for i, req in enumerate(served_order):
        exp = generate_explanation(req, all_contribs[i], feature_cols)
        result.append({**req, "explanation": exp})
    return result
