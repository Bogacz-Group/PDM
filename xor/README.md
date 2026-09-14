# XOR experiments

This folder reproduces Figures 3c–d: firing-rate and spiking predictive dendrite 
networks trained on XOR.

## 1. System requirements

- **OS:** macOS or Linux. Windows should work with the same Python packages; use the Windows venv activation command below.
- **Python:** 3.10 (tested on 3.10.18). 3.11 should also work. Dependencies in `requirements.txt`.
- **Hardware:** laptop CPU.

## 2. Installation

```bash
cd xor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
This should take ~2 minutes on a standard laptop.

## 3. Reproduction

```bash
python firing_rate_xor.py
python spiking_xor.py
python plot_firing_rate_xor.py
python plot_spiking_xor.py
```

This writes `plots/firing_rate_xor.pdf` (Figure 3c) and `plots/spiking_xor.pdf` 
(Figure 3d). To regenerate the figures from the committed CSVs in `results/` 
without retraining, run only the two `plot_*.py` scripts. The expected run time 
on a normal desktop CPU is typically 3–5 minutes.

## 4. Command-line options

The XOR data are generated in the scripts. To use different training settings, pass flags (the values below are the defaults and reproduce the paper's results):

```bash
python firing_rate_xor.py --n-epochs 500 --num-seeds 8 --hidden-size 64 --lr 0.1
python spiking_xor.py --n-steps 20 --n-repeats 10 --hidden-size 128 --theta 0.4
```
