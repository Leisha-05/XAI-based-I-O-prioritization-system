"""
src/scheduler.py — ML-based disk scheduler (upgraded with multi-objective metrics)
"""

from __future__ import annotations
import os
import math
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


def compute_psr(log: List[dict]) -> float:
    """Priority Satisfaction Rate: % of HIGH priority reqs served in first 50%."""
    if not log:
        return 0.0
    high_reqs = [r for r in log if r.get("priority", 0) == 2]
    if not high_reqs:
        return 1.0
    cutoff = len(log) / 2.0
    served_early = sum(1 for r in high_reqs if r.get("position", 0) <= cutoff)
    return round(served_early / len(high_reqs), 4)


def compute_dmr(log: List[dict]) -> float:
    """Deadline Miss Rate: 1 seek unit = 1ms. Miss if cumulative_time > deadline."""
    if not log:
        return 0.0
    cumulative_time = 0
    missed = 0
    for r in log:
        cumulative_time += r.get("seek_time", 0)
        if cumulative_time > r.get("deadline", 9999):
            missed += 1
    return round(missed / len(log), 4)


def compute_fairness_gini(log: List[dict]) -> float:
    """Gini coefficient of cumulative wait times. Lower = fairer."""
    if not log:
        return 0.0
    wait_times = []
    cumulative = 0
    for r in log:
        cumulative += r.get("seek_time", 0)
        wait_times.append(cumulative)
    wait_times = sorted(wait_times)
    n = len(wait_times)
    if n == 1:
        return 0.0
    s = sum(wait_times)
    if s == 0:
        return 0.0
    gini_num = sum((i + 1) * w for i, w in enumerate(wait_times))
    gini = (2 * gini_num) / (n * s) - (n + 1) / n
    return round(max(0.0, min(1.0, gini)), 4)


def compute_regret_timeline(log: List[dict]) -> List[Dict[str, Any]]:
    """Cumulative deadline misses over service positions (bonus viz)."""
    timeline = []
    cumulative_time = 0
    missed_so_far = 0
    for i, r in enumerate(log):
        cumulative_time += r.get("seek_time", 0)
        if cumulative_time > r.get("deadline", 9999):
            missed_so_far += 1
        timeline.append({
            "position":       i + 1,
            "cumulative_time": cumulative_time,
            "missed_so_far":  missed_so_far,
            "request_id":     r.get("id", i + 1),
            "deadline":       r.get("deadline", 9999),
            "priority":       r.get("priority", 0),
        })
    return timeline


def compute_all_metrics(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute PSR, DMR, Fairness, CPS for every algorithm."""
    raw_metrics = {}
    for key, res in results.items():
        if "error" in res or "log" not in res:
            continue
        log = res["log"]
        psr      = compute_psr(log)
        dmr      = compute_dmr(log)
        fairness = compute_fairness_gini(log)
        raw_metrics[key] = {
            "algorithm":       res["algorithm"],
            "total_seek":      res["total_seek"],
            "avg_seek":        res["avg_seek"],
            "psr":             psr,
            "dmr":             dmr,
            "fairness":        fairness,
            "regret_timeline": compute_regret_timeline(log),
        }

    if not raw_metrics:
        return []

    seeks    = [m["total_seek"] for m in raw_metrics.values()]
    max_seek = max(seeks) if max(seeks) > 0 else 1
    ginis    = [m["fairness"] for m in raw_metrics.values()]
    max_gini = max(ginis) if max(ginis) > 0 else 1

    summary = []
    for key, m in raw_metrics.items():
        seek_eff     = 1.0 - (m["total_seek"] / max_seek)
        dl_adherence = 1.0 - m["dmr"]
        psr_score    = m["psr"]
        fairness_sc  = 1.0 - (m["fairness"] / max_gini if max_gini > 0 else 0)

        cps = (
            0.35 * seek_eff +
            0.30 * dl_adherence +
            0.25 * psr_score +
            0.10 * fairness_sc
        )

        summary.append({
            "algorithm":          m["algorithm"],
            "total_seek":         m["total_seek"],
            "avg_seek":           m["avg_seek"],
            "psr":                round(m["psr"] * 100, 2),
            "dmr":                round(m["dmr"] * 100, 2),
            "fairness_gini":      round(m["fairness"], 4),
            "seek_efficiency":    round(seek_eff * 100, 2),
            "deadline_adherence": round(dl_adherence * 100, 2),
            "fairness_score":     round(fairness_sc * 100, 2),
            "cps":                round(cps * 100, 2),
            "regret_timeline":    m["regret_timeline"],
        })

    summary.sort(key=lambda x: x["cps"], reverse=True)
    return summary


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

    summary = compute_all_metrics(results)

    legacy_summary = []
    for key, res in results.items():
        if "error" in res:
            continue
        legacy_summary.append({
            "algorithm":  res["algorithm"],
            "total_seek": res["total_seek"],
            "avg_seek":   res["avg_seek"],
        })
    legacy_summary.sort(key=lambda x: x["total_seek"])

    return {
        "results":        results,
        "summary":        summary,
        "legacy_summary": legacy_summary,
    }
