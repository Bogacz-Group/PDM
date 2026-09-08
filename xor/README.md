# XOR experiments

The code in this folder reproduces Figures 3c–d, training firing-rate and spiking predictive dendrite networks on the XOR task, respectively.

## Set up

```bash
cd xor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
python firing_rate_xor.py
python spiking_xor.py
python plot_firing_rate_xor.py
python plot_spiking_xor.py
```

This writes `plots/firing_rate_xor.pdf` (Figure 3c) and `plots/spiking_xor.pdf` (Figure 3d). To regenerate the figures from the committed CSVs in `results/` without retraining, run only the two `plot_*.py` scripts.
