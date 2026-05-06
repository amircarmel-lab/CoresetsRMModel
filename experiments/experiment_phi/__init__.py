"""Experiment 3: noise robustness of our coreset construction.

Sweeps effective quadruplet-oracle noise phi by mixing a noisy
cheap-embedding oracle with a noise-free expensive-embedding oracle.
Probability beta of routing any given query to the expensive backend
reduces expected effective phi to ``(1 - beta) * phi_real``.

Entry points:
  - :mod:`experiments.experiment3.run_exp3`   — runner (``--test`` flag)
  - :mod:`experiments.experiment3.plot_exp3`  — plot from saved CSV
"""
