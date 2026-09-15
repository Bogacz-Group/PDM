# XOR experiments

This folder reproduces the firing-rate and spiking XOR comparisons in Figures
3c–d and the firing-rate weight trajectories in Supplementary Figure 3c.

| Paper panel | Numerical entry point | Plotting entry point | Original analysis-workspace source |
|---|---|---|---|
| Figure 3c | `firing_rate_xor.py` | `plot_firing_rate_xor.py` | firing-rate XOR experiment |
| Figure 3d | `spiking_xor.py` | `plot_spiking_xor.py` | spiking XOR experiment |
| Supplementary Figure 3c | `weight_trajectories.py` | `plot_weight_trajectories.py` | `experiments/learn/XOR_record_wf_field.ipynb` |

The spiking counterpart of the weight-trajectory panel, Supplementary Figure
3d, is in [`../spiking/`](../spiking/).

The Supplementary Figure 3c source path is pinned to
`YuhangSong/predictive_dendrite` commit
`370b8daebcd6d52e37075f362f29e3a6c59e1ca3` in the repository-level
[`migration_manifest.json`](../migration_manifest.json).

## 1. System requirements

- **Operating system:** tested on 64-bit Linux. The scripts use standard
  Python packages and should also run on macOS. The commands below use a POSIX
  shell.
- **Python:** Python 3.11.14.
- **Hardware:** laptop or desktop CPU; no GPU is required. The firing-rate
  Figure 3c script will use CUDA if PyTorch detects it, but CPU execution is
  supported and is the reproducible baseline.
- **Other software:** Git is required because the open-source experiment
  orchestration dependency is installed from GitHub.
- **Python packages:** `numpy==2.2.6`, `pandas==2.3.3`,
  `matplotlib==3.10.7`, `seaborn==0.13.2`, `scipy==1.16.2`,
  `scikit-learn==1.7.2`, `torch==2.9.0`, `ray==2.53.0`,
  `pyarrow==22.0.0`, `optuna==4.5.0`, and `pytest==8.4.2`.
  The additional dependencies and exact versions are in `requirements.txt`.
- **Experiment orchestration:** the `radas` Python package is installed from
  the public
  [`mini-radas`](https://github.com/YuhangSong/mini-radas) repository at commit
  `8314d91b773db5148b0a90a8442f549e367e9cfb`.

The four XOR examples are generated in the scripts; no dataset download is
required.

## 2. Installation

From a fresh clone, create a self-contained environment for this folder:

```bash
cd PDM/xor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd ..
```

Estimated installation time: **10–20 minutes** on a broadband connection.
This is an estimate rather than a benchmark; downloading the PyTorch and Ray
wheels usually dominates and a warm package cache can be much faster.

Verify the installation from the repository root (estimated runtime: **under
1 minute**):

```bash
python -m pytest xor/tests -q
```

## 3. Reproduction

Run these commands from the repository root with the `xor` environment
activated.

### Figures 3c–d

```bash
python xor/firing_rate_xor.py
python xor/spiking_xor.py
python xor/plot_firing_rate_xor.py
python xor/plot_spiking_xor.py
```

The two training scripts write long-form data to
`xor/results/firing_rate.csv` and `xor/results/spiking.csv`. The plotting
scripts write `xor/plots/firing_rate_xor.pdf` (Figure 3c) and
`xor/plots/spiking_xor.pdf` (Figure 3d).

Estimated runtime on a current desktop CPU: **2–10 minutes** for
`firing_rate_xor.py`, **2–10 minutes** for `spiking_xor.py`, and **under 1
minute** for each plotting script.

### Supplementary Figure 3c

The paper grid contains 81 initial forward-weight pairs (`-1` to `1` in steps
of `0.25`). The mini-radas entry point runs one trajectory per configuration:

```bash
python spiking/run_with_radas.py supp3c --seed 0
```

The direct sequential equivalent is:

```bash
python xor/weight_trajectories.py
python xor/plot_weight_trajectories.py
```

The direct commands write `xor/results/supplementary_figure3c.csv` and
`xor/plots/supplementary_figure3c.{pdf,svg}`. The mini-radas command writes the
CSV, a trial table, PDF, SVG, and `supp3c_run.json` to
`spiking/generated/radas/supp3c/<run-id>/`. Estimated runtime for the full grid:
**1–10 minutes** on a current desktop CPU; plot-only generation is estimated
to take **under 1 minute**.

The source experiment performs 128 complete-XOR-batch optimizer steps. Each
batch update sums the contributions of all four samples, so the same endpoint
is described in the manuscript as `128 x 4 = 512` sample contributions. The
CSV records `optimizer_step` and `sample_updates` separately to make that
distinction explicit.

### Plot-only paths

The committed Figure 3 CSVs can be replotted without retraining. Once the
Supplementary Figure 3c CSV has been generated, it can be handled the same
way:

```bash
python xor/plot_firing_rate_xor.py
python xor/plot_spiking_xor.py
python xor/plot_weight_trajectories.py
```

To replot a mini-radas trajectory table, pass
`--results-path spiking/generated/radas/supp3c/<run-id>/supplementary_figure3c.csv`
together with the desired `--pdf-output` and `--svg-output` paths.

All runtimes in this section are conservative estimates, not measurements.
They vary with CPU, storage, process-startup overhead, and selected options.

## 4. Command-line options

The values below are the defaults used for the paper reproductions. Change
the training duration, network size, or number of seeds for Figures 3c–d:

```bash
python xor/firing_rate_xor.py \
  --n-epochs 500 --num-seeds 8 --hidden-size 64 --lr 0.1
python xor/spiking_xor.py \
  --n-steps 20 --n-repeats 10 --hidden-size 128 --theta 0.4
```

Choose a different initial-weight grid or learning settings for Supplementary
Figure 3c:

```bash
python xor/weight_trajectories.py \
  --grid-min -1 --grid-max 1 --grid-step 0.5 \
  --learning-rate 0.05 --backward-learning-rate 0.05 \
  --num-dataset-iterations 128 --seed 7
```

Run only one initial condition while checking the installation:

```bash
python xor/weight_trajectories.py \
  --initial-wf1 -0.5 --initial-wf2 0.5 \
  --num-dataset-iterations 8 --seed 0
python xor/plot_weight_trajectories.py
```

The same reduced grid can be orchestrated with mini-radas:

```bash
python spiking/run_with_radas.py supp3c \
  --grid-min -1 --grid-max 1 --grid-step 0.5 \
  --iterations 32 --seed 7 --cpus-per-trial 1
```

Use `--results-path` and the plotting scripts' `--output`, `--pdf-output`, or
`--svg-output` flags to redirect files. Use `--no-feature-regions` to omit the
dashed analytical regions from Supplementary Figure 3c. Every script supports
`--help`; `python spiking/run_with_radas.py --help` lists the runner's
`--quick`, `--smoke`, storage, output, and resource controls.
The runner isolates default outputs by target and run ID; the run ID includes
the profile, seed, and a scientific-options hash. Use `--output` when a fixed
human-readable destination is preferable.
