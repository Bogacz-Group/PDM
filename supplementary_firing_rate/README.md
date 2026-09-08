# Supplementary firing-rate experiments

The code in this folder reproduces results of Supplementary Figures 2 (XOR), 4
(linear and nonlinear toy tasks) and 5 (Iris classification). The code is 
JAX-based.

## Installation

```bash
cd supplementary_firing_rate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

From `supplementary_firing_rate/`:

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
