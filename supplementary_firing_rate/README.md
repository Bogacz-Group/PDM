# Supplementary firing-rate experiments

This folder reproduces Supplementary Figures 2 (XOR), 4 (linear and nonlinear
toy tasks), and 5 (Iris classification). The training code is JAX-based.

## 1. System requirements

- **OS:** macOS or Linux. Windows should work with a CPU JAX wheel; use the Windows venv activation command below.
- **Python:** 3.10 (tested on 3.10.18). 3.11 should also work. The `jpc` dependency requires Python ≥ 3.10. Dependencies in `requirements.txt` (`jpc` is installed from GitHub, so `git` and internet access are required).
- **Hardware:** laptop CPU.

## 2. Installation

```bash
cd supplementary_firing_rate
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
This should take 3–5 minutes on a standard laptop.

## 3. Reproduction

```bash
python train_xor.py
python plot_xor.py

python train_toy_linear.py
python plot_toy_linear.py

python train_toy_nonlinear.py
python plot_toy_nonlinear.py

python train_iris.py
python plot_iris.py
```

This writes results under `results/` and figure PDFs next to those results
(Supplementary Figures 2, 4, and 5). On a normal desktop CPU, XOR typically
takes 15–30 minutes. The toy-linear, toy-nonlinear, and Iris scripts typically
take several hours to overnight.

## 4. Instructions for use

To change training hyperparameters, pass flags (the values below are the
defaults and reproduce the paper's results):

```bash
python train_xor.py --n_seeds 5 --n_train_iters 1000 --width 2 --n_hidden 1 --param_lr 5e-3
python train_toy_linear.py --n_seeds 5 --n_train_iters 5000 --param_lrs 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 --slope 3.0 --noise_std 0.5
python train_toy_nonlinear.py --n_seeds 5 --n_train_iters 5000 --param_lrs 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 --width 2 --thetaf 0.5
python train_iris.py --n_seeds 20 --n_train_iters 5000 --widths 16 --param_lrs 0.5 0.05 0.01 0.005 0.001 0.0005 0.0001 --thetafs 0.5 0.9 0.95 0.99 1.0 --cv_folds 5
```
