# FedSAA: Specialty-Aware Adapter Aggregation — a cross-specialty federated LoRA benchmark

Code and results for a benchmark of federated parameter-efficient fine-tuning (LoRA adapters) for
cross-specialty medical image classification on MedMNIST. Each client holds data from a different
medical specialty, and we compare specialty-aware aggregation (FedSAA) with standard federated
baselines.

The paper reports this as an honest benchmark: under our protocol, FedSAA does **not** beat simpler
baselines such as FedProx, FFA-LoRA or local-only training.

## Methods compared
Centralized, Local-only, Specialty oracle, Linear probe, FedAvg, FedProx, FedSA-LoRA, FFA-LoRA,
FlexLoRA, FedSAA, FedSAA (rho = 0), FedSAA-U.

## Repository layout
```
fedbench_features.py   # frozen ResNet-18 feature extraction for the MedMNIST tasks
fedbench_train.py      # federated training (all strategies, presets A/B, linear / non-linear adapters)
fedbench_analyze.py    # tables (analysis/T_*.csv), figures, analysis/summary.md
analysis/              # result tables, figures and summary.md
results/               # runs.jsonl, per-run metrics, selected configs, preset metadata
```

## Reproducing
The experiments were run on Kaggle (CPU, about 10.6 h for 380 main runs plus tuning):
```
python fedbench_features.py
python fedbench_train.py
python fedbench_analyze.py
```
The extracted features (`feats/`) are not stored here because they are large; `fedbench_features.py`
regenerates them.

## Results
Accuracy at round 40 is reported as mean ± SD over seeds, along with macro-F1, AUC and worst-client
accuracy. See `analysis/summary.md` for the full tables.

## Citation
Manuscript under preparation for IEEE Access. Citation details will be added on acceptance.
