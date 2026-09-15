# Recorded best-fit configurations

These JSON files are human- and machine-readable provenance snapshots of the
best configurations printed in the migrated notebooks. The same numbers are
embedded in `fit_stdp.py` and `fit_initial_strength.py`, which makes the default
figure-reproduction path independent of notebook output, Ray checkpoints, and
private infrastructure.

- `fig6i_bi2002.json` and `fig6j_woodin2003.json` were selected using the
  historical observation-index variance weighting documented in
  `../data/README.md`.
- `fig7b_initial_strength.json` was selected from a search in which the mapped
  model-weight range was sampled from `[0.5, 2.0]`. Methods Section H states a
  range of `[0.0, 0.3]`, which cannot contain the reported best value
  `0.6515770896337575`. `fit_initial_strength.py` therefore defaults to the
  executable notebook provenance (`--search-space legacy`) and exposes the
  printed Methods bounds separately (`--manuscript-search-space`).

Fresh fits write a new result JSON and a complete trial CSV under the directory
selected by `--output`; they do not overwrite these provenance snapshots.
