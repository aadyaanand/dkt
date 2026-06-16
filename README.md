# Joint KT + Calibration on ASSISTments 2009

PyTorch implementation of a dual-head LSTM that jointly models **knowledge state** and **metacognitive calibration** from learner interaction sequences.

## What it does

- **Knowledge Tracing (KT):** Predicts whether the learner will answer the next problem correctly (standard DKT-style objective).
- **Calibration modeling:** Predicts the signed gap between self-reported confidence and actual correctness at each timestep.
- **Miscalibration detection:** Flags overconfidence, underconfidence, and calibrated interactions (threshold = 0.3).
- **Intervention routing:** Maps (knowledge estimate, miscalibration label) → one of four intervention types (e.g., targeted explanation, confidence recalibration prompt).

## Model architecture

Shared 2-layer LSTM (hidden size 128) with two heads:
- **KT head:** linear → sigmoid
- **Calibration head:** linear → tanh

**Input features (per timestep):**
1. Skill one-hot
2. Correctness on current skill
3. Confidence (simulated; see below)
4. Normalized response time

A **fair DKT baseline** uses only the first two channels for comparison.

## Dataset

Uses `assistments_2009.csv` (~520K interactions, 110 skills). Because ASSISTments 2009 lacks real confidence ratings, confidence is **simulated** from Beta distributions conditioned on correctness (Beta(8,2) for correct, Beta(2,8) for incorrect) with small Gaussian noise.

Split: **80% train / 10% val / 10% test** by user ID.

## Evaluation

| Metric | Purpose |
|--------|---------|
| AUC | Next-step correctness discrimination |
| RMSE | KT probability error |
| Calibration MSE | Miscalibration residual prediction |
| ECE | Probability calibration of KT predictions |

## Experiments included

- Fair DKT baseline vs. joint model
- Training improvements: 40 epochs, early stopping, LR scheduler, gradient clipping
- λ sweep (0.1, 0.3, 0.5) for calibration loss weight
- No-confidence ablation (confidence channel zeroed)
- Top-10 per-skill miscalibration rates

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `torch`, `scikit-learn`, `jupyter`

## Usage

Place `assistments_2009.csv` in the repo root and run all cells in `dkt_assistments_2009.ipynb`.

Optional: `python run_ablations.py` reruns the λ sweep and no-confidence ablation standalone.

## Key results (test set)

| Model | AUC | RMSE | Cal MSE | ECE |
|-------|-----|------|---------|-----|
| DKT Baseline | 0.9138 | 0.3190 | — | — |
| Joint KT + Calibration | 0.9171 | 0.3161 | 0.0025 | 0.0051 |
