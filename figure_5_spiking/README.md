# Figure 5: spiking predictive dendrites

This directory independently reproduces the numerical panels in Figure 5b-c.
Figure 5a is a conceptual schematic. Source provenance is recorded in
[`provenance.json`](provenance.json).

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

The command writes numerical data and metadata to `results/`, and Figure 5b-c
as PDF and SVG files to `plots/`. Expected runtime is 1-3 minutes.

To rerender existing results without rerunning the simulation:

```bash
python figure5.py --action plot
```

Run the tests with `python -m unittest discover -s tests -v` (normally under
one minute).

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
