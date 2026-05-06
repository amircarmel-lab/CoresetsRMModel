# Coresets for Clustering Using Noisy Comparisons and Few Distance Queries

Code accompanying the NeurIPS submission. All experiments run on CPU.

## Install

Python 3.10+ recommended. The `bico` package pins `numpy<2.0` and will downgrade numpy on install — use a fresh virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets

All datasets are fetched and cached automatically on first run. Cache root is `experiments/cache/` (created on demand). One-time downloads:

| Dataset | Source |
|---|---|
| 20 Newsgroups | `sklearn.datasets.fetch_20newsgroups` |
| US Census 1990 | UCI via `ucimlrepo` (id=116) |
| STL-10 | `torchvision.datasets.STL10` |
| BGE-micro / BGE-large weights | HuggingFace `TaylorAI/bge-micro-v2`, `BAAI/bge-large-en-v1.5` |
| DINOv2 / MobileNetV3 weights | HuggingFace `facebook/dinov2-base`, torchvision |

Subsequent runs hit the cache.

## Reproducing the experiments

Every runner accepts `--test` for a fast smoke run (small `n`, few reps). Use `--test` first to confirm your install is working before launching a full run.

### Experiment 3 — noise robustness (vs. φ)

```bash
# smoke
python -m experiments.experiment_phi.run_exp --semi --test

# full reproduction
for r in 0 1 2 3 4 5 6 7 8 9; do
    python -m experiments.experiment_phi.run_exp --full --rep $r
done
python -m experiments.experiment_phi.run_exp --full --merge

# plots
python -m experiments.experiment_phi.plot_exp
```

### Experiment 4 — distance-query scaling vs. n

```bash
# smoke
python -m experiments.experiment_scaling.run_exp --full --test

# full reproduction
for r in 0 1 2 3 4 5 6 7 8 9; do
    python -m experiments.experiment_scaling.run_exp --full --exp 6 --rep $r
done
python -m experiments.experiment_scaling.run_exp --full --exp 6 --merge

# plots
python -m experiments.experiment_scaling.plot_exp --exp 6
```

### Experiment 5 — cost ratio vs. coreset size

```bash
# smoke
python -m experiments.experiment_strong.run_exp --test

# full reproduction
for r in 0 1 2 3 4 5 6 7 8 9; do
    python -m experiments.experiment_strong.run_exp --rep $r
done
python -m experiments.experiment_strong.run_exp --merge

# plots
python -m experiments.experiment_strong.plot_exp
```

CSV outputs land under each experiment directory; plots land alongside them.

## Repository layout

```
coreset_rm/                   core algorithm (Phase 1 + Phase 2 + oracle abstractions)
experiments/
  baselines/                  BICO, sensitivity sampling, lightweight, uniform, ours
  datasets/                   loaders + caching (20news, census, stl10)
  embeddings/                 BGE (text), DINOv2 + MobileNetV3 (image)
  evaluation/                 strong / weak coreset cost evaluation, NMI
  persistence/                CSV append helper
  experiment_phi/             noise-robustness experiment
  experiment_scaling/         scaling-with-n experiment
  experiment_strong/          cost-vs-size experiment
```
