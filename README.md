# FedSAA: a cross-specialty benchmark of federated low-rank adapters for medical imaging

Code, raw results and analysis for the manuscript *Federated Low-Rank Feature Adapters Across Medical
Imaging Specialties: Specialty Structure, Training Budget, and the Limits of Similarity-Weighted
Aggregation* (in preparation for IEEE Access).

Two simulated federations (Preset A and Preset B) are built from MedMNIST v2, each with five
specialties and three clients per specialty (15 clients). A rank-16 adapter on frozen ResNet-18 features
is shared and every client keeps a private linear classifier. Twelve strategies are compared with
validation-tuned learning rates, 200 rounds, 10 seeds and validation-based early stopping.

## Main findings
- **Whether federation helps depends on the federation.** In Preset A, FedAvg was 0.85 points below
  local-only training, while averaging only within each specialty (specialty oracle) gained +1.98 points.
  In Preset B every federated strategy beat local-only training (+0.64 to +2.49 points).
- **FedSAA does not personalize in practice.** Logged similarities show that adapter products of
  different specialties reach cosine similarity above 0.97 within 40 rounds, so the attention weights
  become uniform and FedSAA reduces to FlexLoRA.
- **Retention must act on updates, not factors.** FedSAA-U (retention in update space) beat FedSAA
  (retention in factor space) by 0.83 points in Preset A, as the factor non-identifiability analysis
  predicts. Rank truncation kept at least 99.7% of the energy.
- **A fixed budget can reverse the ranking.** At round 40 local-only training led every global federated
  strategy in Preset A; at round 200 it trailed all of them.

## Methods compared
Centralized, Local-only, Linear probe, FedAvg, FedProx, Specialty oracle, FlexLoRA, FFA-LoRA,
FedSA-LoRA, FedSAA, FedSAA (rho = 0), FedSAA-U; linear and ReLU (nonlinear) adapters.

## Repository layout
```
fedbench_features.py   # stage 1: build clients, cache frozen ResNet-18 features (needs PyTorch, medmnist)
fedbench_train.py      # stage 2: tuning and main runs, all strategies (NumPy, CPU)
fedbench_analyze.py    # stage 3: tables, tests and figures -> analysis/
notebooks/             # the Kaggle notebook that ran stages 1-3
results/               # runs.jsonl (518 runs: curves, checkpoint metrics, similarity/SVD logs),
                       # runs/*.npz (per-image correctness at rounds 40/100/200), meta_*.json, selected_configs.json
analysis/              # T_*.csv tables, figures, summary.md
manuscript/            # scripts that compute every number, table and figure of the manuscript
LICENSE                # MIT
```

## Reproducing
The experiments ran in a Kaggle CPU notebook (`notebooks/FedSAA_revision_experiments.ipynb`):
```
python fedbench_features.py --out feats --root medmnist_data
python fedbench_train.py --feats feats --out results --stage tune     # repeat until "stage complete"
python fedbench_train.py --feats feats --out results --stage main     # repeat until "stage complete"
python fedbench_analyze.py --feats feats --results results --out analysis
```
The cached features (`feats/`) are not stored because they are large; stage 1 regenerates them.
To rebuild the manuscript tables, figures and Word file from `results/` and `analysis/`, see
[`manuscript/README.md`](manuscript/README.md).

## Data
MedMNIST v2 (https://medmnist.com), CC BY 4.0 (DermaMNIST: CC BY-NC 4.0). Only public, de-identified
data are used. The clients are simulated; nothing here is a clinical validation.

## Citation
Citation details will be added on acceptance.
