# Figure 5: spiking predictive dendrites

This directory independently reproduces the numerical panels in Figure 5b-c.
Figure 5a is a conceptual schematic. The source is
`experiments/spiking/basics.ipynb` at predictive_dendrite commit
[`370b8da`](https://github.com/YuhangSong/predictive_dendrite/commit/370b8daebcd6d52e37075f362f29e3a6c59e1ca3).

## 1. System requirements

Tested on 64-bit Linux with Python 3.11.14; a CPU is sufficient. Exact package
versions are listed in [`requirements.txt`](requirements.txt).

## 2. Installation

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Installation typically takes 5-10 minutes.

## 3. Reproduction

```bash
python figure5.py
```

The command writes the local files `results/figure5.{csv,json}` and Figure
5b-c as PDF and SVG files under `plots/`. These generated outputs are ignored
by Git. Expected runtime is 1-3 minutes.

After running the command once, rerender its local CSV without repeating the
simulation:

```bash
python figure5.py --action plot
```

## 4. Command-line options

The source notebook uses `alpha=0.2`, while the manuscript caption reports
`alpha=0.1`. The historical value is the default; reproduce the caption value
with:

```bash
python figure5.py --parameter-set manuscript
```

Parameters can also be set directly, for example:

```bash
python figure5.py --learning-rate 0.05 --initial-weights 0.5 0.25 \
  --tau-values-ms 0 4 8
```

Use `python figure5.py --help` for all options.
