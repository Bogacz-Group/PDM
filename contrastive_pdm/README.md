# Contrastive PDM

This folder reproduces Figure 4a and Supplementary Figure 6: standard and contrastive predictive dendrite models, compared with two backpropagation baselines, on MNIST and CIFAR-10.

The training code has been adapted and extended from
[Supervised-Predictive-Entropy-Maximization](https://github.com/BariscanBozkurt/Supervised-Predictive-Entropy-Maximization).
That code is MIT licensed, copyright Barışcan Bozkurt.

## 1. System requirements

- **OS:** macOS or Linux. Windows should work with the same Python packages; use the Windows venv activation command below.
- **Python:** 3.10 or later (tested on 3.12). Dependencies in `requirements.txt`.
- **Hardware:** a laptop CPU is enough to regenerate the figures from the saved results.

The committed predictive dendrite runs were trained on TPUs with torch 2.9.0. The backpropagation and fixed-input-weight runs were trained on CUDA with torch 2.10.0+cu126.

## 2. Installation

```bash
cd contrastive_pdm/
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installation should take a few minutes on a standard laptop.

## 3. Reproduction

From `Simulations/`:

```bash
python plot_results.py
```

The script reads `Simulations/Results/`. For each model it keeps the learning-rate setting with the highest final validation accuracy, then writes test-error PDFs to `Simulations/Figures/`:

- `Fig4a_{MNIST,CIFAR10}_test_error.pdf` and `Fig4a_legend.pdf`
- `SuppFig6_{MNIST,CIFAR10}_test_error.pdf` and `SuppFig6_legend.pdf`
To retrain the runs those figures are drawn from, from `Simulations/`:

```bash
bash run_ablation_sweep.sh
```

The sweep retrains only the settings `plot_results.py` selects, not the rest of either learning-rate grid. The predictive dendrite runs use ReLU, with β `0.05` on MNIST and `0.0025` on CIFAR-10.

| Dataset | Model | Selected setting |
|---|---|---|
| MNIST | Contrastive (weak clamp) | base, neural learning rate 1.6 |
| MNIST | Contrastive (full clamp) | low, neural learning rate 1 |
| MNIST | Standard (weak clamp) | base, neural learning rate 0.6 |
| MNIST | Standard | base, neural learning rate 0.6 |
| MNIST | Backpropagation (Adam) | learning rate 0.0005 |
| MNIST | Backpropagation (SGD) | learning rate 0.03 |
| MNIST | Fixed input weights (Adam) | learning rate 0.003 |
| MNIST | Fixed input weights (SGD) | learning rate 0.03 |
| CIFAR-10 | Contrastive (weak clamp) | high, neural learning rate 1.6 |
| CIFAR-10 | Contrastive (full clamp) | high, neural learning rate 0.6 |
| CIFAR-10 | Standard (weak clamp) | base, neural learning rate 1.6 |
| CIFAR-10 | Standard | low, neural learning rate 1 |
| CIFAR-10 | Backpropagation (Adam) | learning rate 5e-05 |
| CIFAR-10 | Backpropagation (SGD) | learning rate 0.001 |
| CIFAR-10 | Fixed input weights (Adam) | learning rate 0.001 |
| CIFAR-10 | Fixed input weights (SGD) | learning rate 0.01 |

Each training script trains 5 seeds for 15 epochs. Downloaded datasets are written to `Simulations/<dataset>/data/` and logs to `Simulations/<dataset>/logs/`. Both directories are gitignored.
