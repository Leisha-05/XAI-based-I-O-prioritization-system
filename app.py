"""
app.py — XAI Disk Scheduler Flask API (upgraded)
NEVER trains. Loads models/model_bundle.pkl only.
Run train_model.py ONCE first.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np

from src.scheduler import (
    load_model, validate_requests, sample_from_dataset,
    run_classic, run_ml_schedule, run_all_algorithms,
    CLASSIC_ALGOS, PRIORITY_DECODE, TYPE_DECODE,
    MODEL_PATH, CSV_PATH, N_CYLINDERS
)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_model(MODEL_PATH)
    return _bundle


def _safe(obj):
    if isinstance(obj, dict):   return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    model_exists   = os.path.exists(MODEL_PATH)
    dataset_exists = os.path.exists(CSV_PATH)
    info = {
        "model_exists": model_exists, "dataset_exists": dataset_exists,
        "ready": model_exists, "model_path": MODEL_PATH,
        "n_cylinders": N_CYLINDERS, "algorithms": CLASSIC_ALGOS + ["ml"],
    }
    if model_exists:
        try:
            b = get_bundle()
            info["model_r2"]    = b.get("r2")
            info["model_rmse"]  = b.get("rmse")
            info["n_train"]     = b.get("n_train")
            info["feature_cols"] = b.get("feature_cols")
            imp_df = b.get("importance_df")
            info["importance"] = imp_df.to_dict("records") if imp_df is not None else []
        except Exception as e:
            info["model_error"] = str(e)
    return jsonify(_safe(info))


@app.route("/api/requests/sample", methods=["POST"])
def sample_requests():
    body = request.get_json() or {}
    n    = min(int(body.get("n", 20)), 100)
    head = int(body.get("head", 100))
    seed = int(body.get("seed", 42))
    try:
        reqs = sample_from_dataset(n=n, seed=seed, head=head)
        for r in reqs:
            r["priority_label"] = PRIORITY_DECODE.get(r["priority"], "?")
            r["type_label"]     = TYPE_DECODE.get(r["type"], "?")
        return jsonify(_safe({"success": True, "requests": reqs}))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/schedule/<algorithm>", methods=["POST"])
def schedule_single(algorithm):
    body      = request.get_json() or {}
    reqs      = validate_requests(body.get("requests", []))
    head      = int(body.get("head", 100))
    direction = body.get("direction", "up")
    if not reqs:
        return jsonify({"success": False, "error": "No requests provided"}), 400
    try:
        if algorithm == "ml":
            result = run_ml_schedule(get_bundle(), reqs, head=head)
        else:
            result = run_classic(algorithm, reqs, head=head, direction=direction)
        return jsonify(_safe({"success": True, **result}))
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/schedule/compare", methods=["POST"])
def schedule_compare():
    body = request.get_json() or {}
    reqs = validate_requests(body.get("requests", []))
    head = int(body.get("head", 100))
    if not reqs:
        return jsonify({"success": False, "error": "No requests provided"}), 400
    try:
        outcome = run_all_algorithms(get_bundle(), reqs, head=head)
        return jsonify(_safe({"success": True, **outcome}))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_single():
    body   = request.get_json() or {}
    bundle = get_bundle()
    model  = bundle["model"]
    feature_cols = bundle["feature_cols"]
    import pandas as pd
    try:
        head = int(body.get("head", 100))
        req  = validate_requests([body])[0]
        req["seek_distance"] = abs(req["cylinder"] - head)
        X     = pd.DataFrame([[req[f] for f in feature_cols]], columns=feature_cols)
        score = float(model.predict(X)[0])
        from src.xai import compute_contributions, generate_explanation
        contribs  = compute_contributions(model, feature_cols, X)[0]
        req["ml_score"] = score
        exp = generate_explanation(req, contribs, feature_cols)
        return jsonify(_safe({
            "success": True, "score": round(score, 5),
            "explanation": exp,
            "feature_values": {f: req[f] for f in feature_cols},
        }))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  XAI Disk Scheduler")
    print(f"  Model: {'OK' if os.path.exists(MODEL_PATH) else 'MISSING - run train_model.py'}")
    print("=" * 50)
    app.run(debug=True, port=5000)
