"""
train_model.py — Offline training script for XAI Disk Scheduler
Run ONCE before starting app.py:
    python train_model.py

Generates:
    data/dataset.csv
    models/model_bundle.pkl
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
CSV_PATH   = os.path.join(DATA_DIR, "dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model_bundle.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────
N_REQUESTS     = 100_000
N_CYLINDERS    = 200       # disk cylinders 0–199
SEED           = 42
PRIORITY_MAP   = {"Low": 0, "Medium": 1, "High": 2}
TYPE_MAP       = {"Read": 0, "Write": 1}
FEATURE_COLS   = ["cylinder", "seek_distance", "deadline", "priority", "size", "type"]


# ── Dataset generation ────────────────────────────────────────

def generate_dataset(n=N_REQUESTS, seed=SEED):
    """
    Generate realistic synthetic disk I/O request dataset.

    Each request has:
      cylinder     : target cylinder on disk (0–199)
      seek_distance: estimated head movement cost (filled relative to prev request)
      deadline     : urgency in ms (lower = more urgent)
      priority     : Low=0 / Medium=1 / High=2
      size         : request size in KB (4–1024)
      type         : Read=0 / Write=1

    Target (importance_score):
      A weighted composite of seek cost, urgency, and priority.
      Higher score → higher scheduling priority.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    cylinders     = rng.integers(0, N_CYLINDERS, size=n)
    # seek_distance relative to previous request (simulates realistic queue)
    head_positions = np.roll(cylinders, 1)
    head_positions[0] = N_CYLINDERS // 2
    seek_distances = np.abs(cylinders - head_positions).astype(float)

    deadlines  = rng.integers(5, 500, size=n).astype(float)   # ms
    priorities = rng.integers(0, 3, size=n)                   # 0,1,2
    sizes      = rng.integers(4, 1025, size=n).astype(float)  # KB
    types      = rng.integers(0, 2, size=n)                   # 0=Read,1=Write

    # ── Scoring formula ──────────────────────────────────────
    # Higher seek → more costly to skip → higher priority
    seek_norm     = seek_distances / (N_CYLINDERS - 1)
    # Lower deadline → more urgent → higher priority
    urgency_norm  = 1.0 - (deadlines / 500.0)
    # Priority contribution
    priority_norm = priorities / 2.0
    # Size: larger reads are more valuable, smaller writes more urgent
    size_norm     = sizes / 1024.0

    importance_score = (
        0.35 * seek_norm
        + 0.30 * urgency_norm
        + 0.25 * priority_norm
        + 0.10 * size_norm
    )
    # Add mild noise to avoid perfect rule-memorisation
    noise = rng.normal(0, 0.02, size=n)
    importance_score = np.clip(importance_score + noise, 0.0, 1.0)

    df = pd.DataFrame({
        "cylinder":      cylinders,
        "seek_distance": seek_distances.astype(int),
        "deadline":      deadlines.astype(int),
        "priority":      priorities,
        "size":          sizes.astype(int),
        "type":          types,
        "importance_score": importance_score.round(6),
    })
    return df


# ── Training ──────────────────────────────────────────────────

def train(df):
    X = df[FEATURE_COLS]
    y = df["importance_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_leaf=20,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    # Feature importance (built-in MDI)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importances,
        "pct":        (importances * 100).round(2),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"\n  Feature importances:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["pct"] / 2)
        print(f"    {row['feature']:15s} {row['pct']:5.1f}%  {bar}")

    bundle = {
        "model":          model,
        "feature_cols":   FEATURE_COLS,
        "importance_df":  importance_df,
        "rmse":           round(rmse, 6),
        "r2":             round(r2, 6),
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "n_estimators":   model.n_estimators,
        "n_cylinders":    N_CYLINDERS,
        "priority_map":   PRIORITY_MAP,
        "type_map":       TYPE_MAP,
    }
    return bundle


# ── XAI: Manual SHAP-like contributions ──────────────────────
# We use the model's staged_predict to compute each tree's marginal
# contribution, then group by feature using the tree structure.
# This gives a fast, library-free approximation of SHAP values.

def compute_shap_approx(model, X_sample):
    """
    Compute approximate SHAP values via tree path contribution method.
    Returns ndarray of shape (n_samples, n_features).
    """
    n_samples  = len(X_sample)
    n_features = X_sample.shape[1]
    contribs   = np.zeros((n_samples, n_features))

    X_arr = X_sample.values if hasattr(X_sample, "values") else X_sample

    for tree_wrapper in model.estimators_.ravel():
        tree  = tree_wrapper.tree_
        lr    = model.learning_rate

        # For each sample, walk the decision path
        node_indicators = tree.decision_path(X_arr.astype(np.float32))

        for i in range(n_samples):
            node_path = node_indicators.indices[
                node_indicators.indptr[i]:node_indicators.indptr[i + 1]
            ]
            prev_val = tree.value[node_path[0], 0, 0]
            for node in node_path[1:]:
                curr_val = tree.value[node, 0, 0]
                feat     = tree.feature[node_path[np.where(node_path == node)[0][0] - 1]]
                if feat >= 0:
                    contribs[i, feat] += lr * (curr_val - prev_val)
                prev_val = curr_val

    return contribs


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  XAI Disk Scheduler — Offline Training")
    print("=" * 55)

    print(f"\n[1/3] Generating dataset ({N_REQUESTS:,} rows)…")
    df = generate_dataset()
    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved → {CSV_PATH}")
    print(f"  Score range: {df['importance_score'].min():.3f} – {df['importance_score'].max():.3f}")
    print(f"  Mean score : {df['importance_score'].mean():.3f}")

    print(f"\n[2/3] Training GradientBoostingRegressor…")
    bundle = train(df)

    print(f"\n[3/3] Saving model bundle → {MODEL_PATH}")
    joblib.dump(bundle, MODEL_PATH)

    print("\n✓ Done. Start the app with:  python app.py")
    print("=" * 55)
