# Contrastive PDM

This folder reproduces Figure 4a and Supplementary Figure 6, comparing standard 
and contrastive predictive dendrite models with two backpropagation baselines, 
on MNIST and CIFAR-10 classification. The training code has been adapted and 
extended from [Supervised-Predictive-Entropy-Maximization](https://github.com/BariscanBozkurt/Supervised-Predictive-Entropy-Maximization), which is MIT licensed (copyright 
Barışcan Bozkurt).

## 1. System requirements

- **OS:** macOS or Linux. Windows should work with the same Python packages; use the Windows venv activation command below.
- **Python:** 3.10 or later (tested on 3.12). Dependencies in `requirements.txt`.
- **Hardware:** a laptop CPU is enough to regenerate the figures from the saved results.

The committed predictive dendrite runs were trained on TPUs with torch 2.9.0, 
while the backpropagation and fixed-input-weight runs were trained on CUDA with 
torch 2.10.0+cu126. See below for more details.

## 2. Installation

```bash
cd contrastive_pdm/
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installation should take a few minutes on a standard laptop.

## 3. Reproduction

To reproduce the plots of Figure 4a and Supplementary Figure 6, from 
`Simulations/` run:

```bash
python plot_results.py
```

This selects the saved runs with the highest validation accuracy across learning 
rates and plots the figures in `Simulations/Figures/`. To retrain the runs those 
figures are drawn from, from `Simulations/`:

```bash
bash run_bp_sweep.sh
bash run_pdm_best_configs.sh
```

`run_bp_sweep.sh` retrains the backprop and fixed-input-weight learning-rate 
grids, one run at a time, while `run_pdm_best_configs.sh` retrains only the 
eight best predictive dendrite model variants (across learning rates) in the 
table below. A full rerun from scratch should take about 12 hours for 
`run_bp_sweep.sh` and about a day for `run_pdm_best_configs.sh` on a CPU. 

In the table below, "low", "base", and "high" refer to the three weight learning 
rates used in each training script (`--lr-config-idx` 0, 1, and 2). Feedforward 
(`ff`) and feedback (`fb`) are the two layer rates of the selected setting. For 
full clamp, the unused feedback rate is omitted.

| Dataset | Model | Selected setting |
|---|---|---|
| MNIST | Contrastive (weak clamp) | base (ff 0.05, 0.035; fb 0.0075, 0.0075), activity lr 1.6 |
| MNIST | Contrastive (full clamp) | low (ff 0.03, 0.02; fb 0.02), activity lr 1 |
| MNIST | Standard (weak clamp) | base (ff 0.05, 0.05; fb 0.001, 0.005), activity lr 0.6 |
| MNIST | Standard | base (ff 0.005, 0.005; fb 0.001, 0.001), activity lr 0.6 |
| MNIST | Backpropagation (Adam) | learning rate 0.0005 |
| MNIST | Backpropagation (SGD) | learning rate 0.03 |
| MNIST | Fixed input weights (Adam) | learning rate 0.003 |
| MNIST | Fixed input weights (SGD) | learning rate 0.03 |
| CIFAR-10 | Contrastive (weak clamp) | high (ff 0.006, 0.003; fb 0.003), activity lr 1.6 |
| CIFAR-10 | Contrastive (full clamp) | high (ff 0.045, 0.022; fb 0.015), activity lr 0.6 |
| CIFAR-10 | Standard (weak clamp) | base (ff 0.01, 0.01; fb 0.0001, 0.01), activity lr 1.6 |
| CIFAR-10 | Standard | low (ff 0.0025, 0.0025; fb 0.0005, 0.0005), activity lr 1 |
| CIFAR-10 | Backpropagation (Adam) | learning rate 5e-05 |
| CIFAR-10 | Backpropagation (SGD) | learning rate 0.001 |
| CIFAR-10 | Fixed input weights (Adam) | learning rate 0.001 |
| CIFAR-10 | Fixed input weights (SGD) | learning rate 0.01 |

Each training script trains 5 seeds for 15 epochs. Downloaded datasets are 
written to `Simulations/<dataset>/data/` and logs to `Simulations/<dataset>/logs/`. 
Both directories are gitignored.

`run_pdm_sweep_tpu.sh` is how those predictive-dendrite sweeps were run. On a TPU it 
launches up to four chips at once and covers the 3×3 learning-rate grid 
(`--lr-config-idx` `{0,1,2}` and `--neural-lr-start` `{0.6,1.0,1.6}`), with ReLU, 
no gating, and β `0.05` on MNIST and `0.0025` on CIFAR-10:

```bash
bash run_pdm_sweep_tpu.sh --dataset MNIST
bash run_pdm_sweep_tpu.sh --dataset CIFAR10
```

The script took no more than 48 hours for each dataset.
