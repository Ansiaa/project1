# Production Vision Inspection (VisA: pcb4 + cashew)

A production-style anomaly inspection pipeline built on VisA dataset (pcb4, cashew).
Includes OpenCV preprocessing (ROI crop/align, CLAHE), anomaly detection (AE baseline → PatchCore),
heatmap/overlay visualization, connected-components defect measurement, and MLflow experiment tracking.
Planned deliverables: drift evaluation (weak/mid/strong), ablation (ROI/CLAHE), FastAPI + Docker serving,
and VLM overlay-grounded JSON explanation with guardrails.

---

## Current Evidence (already working)
PatchCore overlay grids:
- pcb4: `./patchcore_overlay_grid_pcb4.png`
- cashew: `./patchcore_overlay_grid_cashew.png`

Artifacts (generated outputs) live in:
- `./artifacts/` (heatmaps / overlays / masks / preds / models / runs)

MLflow tracking:
- `./mlruns/` + `./mlflow.db`

---

## Project Structure
Root:
- `data/`        : dataset location (VisA root should be placed here)
- `artifacts/`   : generated outputs (heatmaps, overlays, masks, predictions, exported models)
- `mlruns/`      : MLflow artifacts
- `mlflow.db`    : MLflow sqlite backend
- `src/`         : source code
- `requirements.txt`

Source tree:
- `src/preprocess/`
  - `run.py`        : preprocess entry (ROI/CLAHE pipeline)
  - `pipeline.py`   : preprocess pipeline orchestration
  - `roi.py`        : ROI crop/align
  - `clahe.py`      : CLAHE
  - `stats.py`      : preprocessing stats
- `src/train/`
  - `ae_train.py`
  - `patchcore_train.py`
  - `patchcore_eval_mlflow.py`
  - `patchcore_export.py`
- `src/infer/`
  - `ae_infer.py`
  - `patchcore_infer.py`
- `src/postprocess/`
  - `cc.py`         : connected components / defect measurement
- `src/vis/`
  - `overlay.py`    : overlay visualization
- `src/utils/`
  - `io.py`         : basic IO helpers

---

## Environment Setup (Windows)
```bash
cd D:\project1
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt


