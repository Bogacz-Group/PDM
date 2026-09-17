# XOR experiments

This folder reproduces Figures 3c–d and the firing-rate weight trajectories in
Supplementary Figure 3c.

| Paper panel | Numerical entry point | Plotting entry point |
|---|---|---|
| Figure 3c | `firing_rate_xor.py` | `plot_firing_rate_xor.py` |
| Figure 3d | `spiking_xor.py` | `plot_spiking_xor.py` |
| Supplementary Figure 3c | `weight_trajectories.py` | `plot_weight_trajectories.py` |

The original notebook path and pinned source revision are recorded in
[`../migration_manifest.json`](../migration_manifest.json).

## 1. System requirements

- 64-bit Linux and Python 3.11.14.
- A CPU is sufficient; no GPU is required.
- Exact package versions are pinned in [`requirements.txt`](requirements.txt),
  including public
  [`mini-radas`](https://github.com/YuhangSong/mini-radas) at commit
  `8314d91b773db5148b0a90a8442f549e367e9cfb`.

## 2. Installation

```bash
cd PDM/xor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd ..
```

Estimated installation time: **10–20 minutes**. To verify the installation
(typically **under 1 minute**), run from the repository root:

```bash
python -m pytest xor/tests -q
```

## 3. Reproduction

Run commands from the repository root with the environment activated.

| Panel | Commands | Main output | Estimated runtime |
|---|---|---|---|
| Figure 3c | `python xor/firing_rate_xor.py`<br>`python xor/plot_firing_rate_xor.py` | `xor/results/firing_rate.csv`, `xor/plots/firing_rate_xor.pdf` | 2–10 min + under 1 min for plotting |
| Figure 3d | `python xor/spiking_xor.py`<br>`python xor/plot_spiking_xor.py` | `xor/results/spiking.csv`, `xor/plots/spiking_xor.pdf` | 2–10 min + under 1 min for plotting |
| Supplementary Figure 3c | `python spiking/run_with_radas.py supp3c --seed 0` | CSV, trials, PDF/SVG, and metadata under `spiking/generated/radas/supp3c/<run-id>/` | 1–10 min |

The direct sequential alternative for Supplementary Figure 3c is:

```bash
python xor/weight_trajectories.py
python xor/plot_weight_trajectories.py
```

These write `xor/results/supplementary_figure3c.csv` and
`xor/plots/supplementary_figure3c.{pdf,svg}`; estimated runtimes are **1–10
minutes** and **under 1 minute**, respectively.

To replot the committed Figure 3 data, run only the two `plot_*_xor.py`
commands above.

## 4. Command-line options

```bash
# Change training settings for Figures 3c–d
python xor/firing_rate_xor.py \
  --n-epochs 500 --num-seeds 8 --hidden-size 64 --lr 0.1
python xor/spiking_xor.py \
  --n-steps 20 --n-repeats 10 --hidden-size 128 --theta 0.4

# Change the Supplementary Figure 3c grid and learning settings
python xor/weight_trajectories.py \
  --grid-min -1 --grid-max 1 --grid-step 0.5 \
  --learning-rate 0.05 --num-dataset-iterations 128 --seed 7

# Reduced mini-radas run
python spiking/run_with_radas.py supp3c --quick --seed 7
```

The source experiment uses 128 whole-dataset optimizer steps, equivalent to
512 sample contributions; the CSV records both counts. Every entry point
supports `--help` for the full option list.
