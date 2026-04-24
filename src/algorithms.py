"""
src/algorithms.py — Classic disk scheduling algorithms
All algorithms operate on a list of Request dicts:
    { id, cylinder, seek_distance, deadline, priority, size, type }
And return:
    {
        algorithm: str,
        order: [request_ids in service order],
        seek_sequence: [cylinder values visited],
        total_seek: int,
        avg_seek: float,
        log: [per-request records],
    }
"""

from __future__ import annotations
from typing import List, Dict, Any
import copy


def _make_result(algo_name: str, log: list, head_start: int) -> Dict[str, Any]:
    seek_seq = [head_start] + [r["cylinder"] for r in log]
    total    = sum(r["seek_time"] for r in log)
    return {
        "algorithm":   algo_name,
        "order":       [r["id"] for r in log],
        "seek_sequence": seek_seq,
        "total_seek":  total,
        "avg_seek":    round(total / len(log), 2) if log else 0,
        "log":         log,
    }


def _entry(req: dict, pos: int, seek: int) -> dict:
    return {
        "position":  pos,
        "id":        req["id"],
        "cylinder":  req["cylinder"],
        "seek_time": seek,
        "deadline":  req["deadline"],
        "priority":  req["priority"],
        "size":      req["size"],
        "type":      req["type"],
    }


# ── FCFS ──────────────────────────────────────────────────────

def fcfs(requests: List[dict], head: int = 100, **_) -> Dict[str, Any]:
    """First Come First Served — serve in arrival order."""
    log = []
    cur = head
    for i, req in enumerate(requests):
        seek = abs(req["cylinder"] - cur)
        log.append(_entry(req, i + 1, seek))
        cur = req["cylinder"]
    return _make_result("FCFS", log, head)


# ── SSTF ──────────────────────────────────────────────────────

def sstf(requests: List[dict], head: int = 100, **_) -> Dict[str, Any]:
    """Shortest Seek Time First — always pick the closest cylinder."""
    remaining = list(copy.deepcopy(requests))
    log = []
    cur = head
    while remaining:
        closest = min(remaining, key=lambda r: abs(r["cylinder"] - cur))
        seek = abs(closest["cylinder"] - cur)
        log.append(_entry(closest, len(log) + 1, seek))
        cur = closest["cylinder"]
        remaining.remove(closest)
    return _make_result("SSTF", log, head)


# ── SCAN (elevator) ───────────────────────────────────────────

def scan(requests: List[dict], head: int = 100,
         direction: str = "up", max_cylinder: int = 199, **_) -> Dict[str, Any]:
    """
    SCAN (Elevator) — sweeps one direction, then reverses.
    direction: 'up' (toward 199) or 'down' (toward 0).
    """
    reqs = sorted(requests, key=lambda r: r["cylinder"])
    cylinders = [r["cylinder"] for r in reqs]

    left  = [r for r in reqs if r["cylinder"] < head]
    right = [r for r in reqs if r["cylinder"] >= head]

    order = []
    if direction == "up":
        order = right + list(reversed(left))
    else:
        order = list(reversed(left)) + right

    log = []
    cur = head
    for req in order:
        seek = abs(req["cylinder"] - cur)
        log.append(_entry(req, len(log) + 1, seek))
        cur = req["cylinder"]
    return _make_result("SCAN", log, head)


# ── C-SCAN ────────────────────────────────────────────────────

def cscan(requests: List[dict], head: int = 100,
          max_cylinder: int = 199, **_) -> Dict[str, Any]:
    """
    Circular SCAN — sweeps up, jumps to 0 without counting, sweeps up again.
    Provides more uniform wait times than SCAN.
    """
    reqs  = sorted(requests, key=lambda r: r["cylinder"])
    right = [r for r in reqs if r["cylinder"] >= head]
    left  = [r for r in reqs if r["cylinder"] < head]
    order = right + left   # up then wrap

    log = []
    cur = head
    for req in order:
        seek = abs(req["cylinder"] - cur)
        log.append(_entry(req, len(log) + 1, seek))
        cur = req["cylinder"]
    return _make_result("C-SCAN", log, head)


# ── LOOK ──────────────────────────────────────────────────────

def look(requests: List[dict], head: int = 100,
         direction: str = "up", **_) -> Dict[str, Any]:
    """
    LOOK — like SCAN but only goes as far as the outermost request,
    not to the physical end of the disk.
    """
    left  = sorted([r for r in requests if r["cylinder"] < head],
                   key=lambda r: r["cylinder"])
    right = sorted([r for r in requests if r["cylinder"] >= head],
                   key=lambda r: r["cylinder"])

    if direction == "up":
        order = right + list(reversed(left))
    else:
        order = list(reversed(left)) + right

    log = []
    cur = head
    for req in order:
        seek = abs(req["cylinder"] - cur)
        log.append(_entry(req, len(log) + 1, seek))
        cur = req["cylinder"]
    return _make_result("LOOK", log, head)


# ── C-LOOK ────────────────────────────────────────────────────

def clook(requests: List[dict], head: int = 100, **_) -> Dict[str, Any]:
    """
    C-LOOK — LOOK version of C-SCAN. Goes to the farthest request, jumps
    back to the closest, continues upward. No wasted travel to disk ends.
    """
    left  = sorted([r for r in requests if r["cylinder"] < head],
                   key=lambda r: r["cylinder"])
    right = sorted([r for r in requests if r["cylinder"] >= head],
                   key=lambda r: r["cylinder"])
    order = right + left

    log = []
    cur = head
    for req in order:
        seek = abs(req["cylinder"] - cur)
        log.append(_entry(req, len(log) + 1, seek))
        cur = req["cylinder"]
    return _make_result("C-LOOK", log, head)


# ── Registry ──────────────────────────────────────────────────

ALGORITHM_MAP = {
    "fcfs":   fcfs,
    "sstf":   sstf,
    "scan":   scan,
    "cscan":  cscan,
    "look":   look,
    "clook":  clook,
}


def run_algorithm(name: str, requests: List[dict], head: int = 100,
                  direction: str = "up") -> Dict[str, Any]:
    fn = ALGORITHM_MAP.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown algorithm: {name!r}. Choose from {list(ALGORITHM_MAP)}")
    return fn(requests, head=head, direction=direction)
