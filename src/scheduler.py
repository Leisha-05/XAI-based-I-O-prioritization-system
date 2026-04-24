"""
src/scheduler.py — ML-based disk scheduler (upgraded)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any

from src.algorithms import run_algorithm, ALGORITHM_MAP
from src.xai import explain_schedule

MODEL_PATH = os.path.join("models", "model_bundle.pkl")
CSV_PATH   = os.path.join("data", "dataset.csv")

PRIORITY_DECODE = {0: "Low", 1: "Medium", 2: "High"}
TYPE_DECODE     = {0: "Read", 1: "Write"}
PRIORITY_MAP    = {"Low": 0, "Medium": 1, "High": 2}
TYPE_MAP        = {"Read": 0, "Write": 1}
N_CYLINDERS     = 200


def load_model(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path!r}. Run train_model.py first.")
    return joblib.load(path)


def validate_requests(raw: list) -> List[dict]:
    out = []
    for i, r in enumerate(raw):
        req = dict(r)
        req["id"] = req.get("id", i + 1)
        p = req.get("priority", 1)
        if isinstance(p, str):
            p = PRIORITY_MAP.get(p, 1)
        req["priority"] = int(p)
        t = req.get("type", 0)
        if isinstance(t, str):
            t = TYPE_MAP.get(t, 0)
        req["type"] = int(t)
        req["cylinder"]  = int(req.get("cylinder", 100))
        req["deadline"]  = int(req.get("deadline", 200))
        req["size"]      = int(req.get("size", 64))
        out.append(req)
    return out


def enrich_with_seek(requests: List[dict], head: int) -> List[dict]:
    enriched = []
    for req in requests:
        r = dict(req)
        r["seek_distance"] = abs(r["cylinder"] - head)
        enriched.append(r)
    return enriched


def sample_from_dataset(n: int = 20, seed: int = 42, head: int = 100) -> List[dict]:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH!r}.")
    df = pd.read_csv(CSV_PATH).sample(n=n, random_state=seed).reset_index(drop=True)
    reqs = []
    for i, row in df.iterrows():
        reqs.append({
            "id":            i + 1,
            "cylinder":      int(row["cylinder"]),
            "seek_distance": int(abs(row["cylinder"] - head)),
            "deadline":      int(row["deadline"]),
            "priority":      int(row["priority"]),
            "size":          int(row["size"]),
            "type":          int(row["type"]),
        })
    return reqs


def run_classic(algorithm: str, requests: List[dict],
                head: int = 100, direction: str = "up") -> Dict[str, Any]:
    reqs = enrich_with_seek(requests, head)
    result = run_algorithm(algorithm, reqs, head=head, direction=direction)
    for entry in result["log"]:
        entry["priority_label"] = PRIORITY_DECODE.get(entry["priority"], "?")
        entry["type_label"]     = TYPE_DECODE.get(entry["type"], "?")
    return result


def run_ml_schedule(bundle: dict, requests: List[dict],
                    head: int = 100) -> Dict[str, Any]:
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]

    remaining  = enrich_with_seek(requests, head)
    served     = []
    cur_head   = head
    total_seek = 0

    while remaining:
        for r in remaining:
            r["seek_distance"] = abs(r["cylinder"] - cur_head)
        X = pd.DataFrame([{f: r[f] for f in feature_cols} for r in remaining])
        scores   = model.predict(X)
        best_idx = int(np.argmax(scores))
        best     = dict(remaining[best_idx])
        best["ml_score"]  = round(float(scores[best_idx]), 5)
        best["seek_time"] = best["seek_distance"]
        total_seek       += best["seek_time"]
        cur_head          = best["cylinder"]
        served.append(best)
        remaining.pop(best_idx)

    served_with_xai = explain_schedule(model, feature_cols, served)
    seek_seq = [head] + [r["cylinder"] for r in served_with_xai]
    for i, r in enumerate(served_with_xai):
        r["position"]       = i + 1
        r["priority_label"] = PRIORITY_DECODE.get(r["priority"], "?")
        r["type_label"]     = TYPE_DECODE.get(r["type"], "?")

    return {
        "algorithm":     "ML Scheduler",
        "order":         [r["id"] for r in served_with_xai],
        "seek_sequence": seek_seq,
        "total_seek":    total_seek,
        "avg_seek":      round(total_seek / len(served_with_xai), 2) if served_with_xai else 0,
        "log":           served_with_xai,
    }


CLASSIC_ALGOS = list(ALGORITHM_MAP.keys())


def run_all_algorithms(bundle: dict, requests: List[dict],
                        head: int = 100) -> Dict[str, Any]:
    results = {}
    for algo in CLASSIC_ALGOS:
        try:
            results[algo] = run_classic(algo, requests, head=head)
        except Exception as e:
            results[algo] = {"error": str(e)}
    try:
        results["ml"] = run_ml_schedule(bundle, requests, head=head)
    except Exception as e:
        results["ml"] = {"error": str(e)}

    summary = []
    for key, res in results.items():
        if "error" in res:
            continue
        summary.append({
            "algorithm":  res["algorithm"],
            "total_seek": res["total_seek"],
            "avg_seek":   res["avg_seek"],
        })
    summary.sort(key=lambda x: x["total_seek"])
    return {"results": results, "summary": summary}
