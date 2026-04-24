# XAI Disk Scheduler

A practical, explainable, ML-based disk scheduling simulator.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model ONCE (generates data/ and models/)
python train_model.py

# 3. Run the web app
python app.py
# → Open http://localhost:5000
```

## Project Structure

```
/
  app.py              Flask API (no training, loads model only)
  train_model.py      Offline training script (run once)
  requirements.txt
  /data
    dataset.csv       100k synthetic disk I/O requests
  /models
    model_bundle.pkl  Trained GradientBoostingRegressor + metadata
  /src
    scheduler.py      ML scheduler + queue management
    algorithms.py     FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK
    xai.py            Feature contribution + explanation engine
  /static
    index.html        Full single-page frontend
```

## Algorithms

| Algorithm | Strategy |
|-----------|----------|
| FCFS | Arrival order |
| SSTF | Closest cylinder first |
| SCAN | Elevator sweep |
| C-SCAN | Circular elevator |
| LOOK | SCAN without disk ends |
| C-LOOK | C-SCAN without disk ends |
| ML | GradientBoosting importance score |

## XAI

The ML scheduler uses tree-path feature contribution analysis
(approximated SHAP) to explain every scheduling decision.
