Federated Learning (Binary) — Robust Aggregation with Anomaly Detection

Overview
Train a federated model for a two‑class image classification task (e.g., CIFAR‑10 subset), robust to Byzantine clients via anomaly detection and trust‑weighted aggregation. Includes baselines (FedAvg, Krum, Median, Trimmed Mean), attacks (label flip, poison, backdoor), and comprehensive evaluation.

Milestones

Baseline FedAvg on clean data.

Implement label flip, data poisoning, and backdoor with triggered test set.

Add anomaly features (statistical + deep/influence) and calibrate thresholds.

Integrate trust EMA and weight mapping; enable recovery and decay.

Add safety prior (blend with Median/Trimmed Mean when dispersion spikes).

Run sweeps over adversary fraction, participation, non‑IID severity, and attack rates.

Produce plots/tables for accuracy/AUROC, ASR, TPR/FPR, overhead, and latency.

Project structure

data/: two‑class filtering, partitions, split indices.

models/: small CNN/ResNet with binary head.

fl/: server, client, aggregators (FedAvg, trust, robust).

attacks/: label flip, poison, backdoor triggers.

anomaly/: statistical/deep features, scoring, thresholds.

trust/: EMA update, bounds, weight mapping.

eval/: metrics and plots.

configs/: base.yaml and overrides.

scripts/: bootstrap, run, sweeps.

results/: results.jsonl, figures, tables.

logs/: optional detailed logs/checkpoints.

Quickstart

Environment:

python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

Setup:

mkdir -p data models fl attacks anomaly trust eval scripts configs results logs

for d in data models fl attacks anomaly trust eval; do touch $d/init.py; done

Prepare data:

python data/filter_two_class.py --classes 1 9 --val-frac 0.02 --out data/splits.json --seed 42

python data/partition.py --splits data/splits.json --scheme iid --num-clients 50 --out data/partition_iid.json --seed 42

Run baseline:

python scripts/run_exp.py --config configs/base.yaml

Reproducibility

Fixed seeds for filtering, partitioning, sampling, and initialization.

Persisted splits and partitions with hashes referenced in run metadata.

Config hashes recorded in results JSONL lines; per‑round metrics and decisions logged.

Version pinning: store pip freeze with each run; periodic checkpoints for resume/audit.

